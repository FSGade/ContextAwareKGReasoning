#!/usr/bin/env python3
"""
RQ2 Pipeline Orchestrator

Manages the complete RQ2 pipeline with Slurm job dependencies.

Two modes:
  --submit only:      Submit all jobs with Slurm dependency chains (fast, fire-and-forget)
  --submit --wait:    Submit each step, wait for completion, then submit next step
                      (no Slurm dependencies needed since orchestrator serializes steps)

Pipeline:
1. Preprocess (1 job)
2. Aggregate (4 jobs - one per tissue)
3. Inference (8 jobs - 4 tissues × 2 hops)
4. Compare (2 jobs - one per comparison)
5. Enrichment (4 jobs - 2 comparisons × 2 hops)
6. Permutation (N×2 jobs - arrays, per-triple null distributions)
7. Compute P-values (2 jobs - aggregate permutations → per-triple p/q values)
8. Reports (4 jobs - 2 comparisons × 2 hops)
9. Volcano Plots (4 jobs - 2 comparisons × 2 hops)
10. Result Visualizations (2 jobs - 2 comparisons, covers both hops)

Usage:
    python orchestrator.py --config config.yaml --submit [--wait] [--step STEP]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set

from utils import load_config

import yaml

# Force unbuffered output — critical when running inside Slurm jobs
# where buffered output gets lost if job is killed
import functools
print = functools.partial(print, flush=True)


def create_slurm_script(script_name: str, python_cmd: str, job_name: str,
                        output_dir: Path, cpus: int = 1, memory: str = "16G",
                        time_limit: str = "01:00:00", partition: str = None,
                        array: str = None, log_subdir: str = None) -> Path:
    """Create a Slurm submission script."""
    scripts_dir = output_dir / 'slurm_scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = output_dir / 'logs'
    if log_subdir:
        logs_dir = logs_dir / log_subdir
    logs_dir.mkdir(parents=True, exist_ok=True)

    script_path = scripts_dir / f'{script_name}.sh'

    directives = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={logs_dir}/{job_name}_%j.out",
        f"#SBATCH --error={logs_dir}/{job_name}_%j.err",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --time={time_limit}",
    ]

    if partition:
        directives.append(f"#SBATCH --partition={partition}")

    if array:
        directives.append(f"#SBATCH --array={array}")
        directives[2] = f"#SBATCH --output={logs_dir}/{job_name}_%A_%a.out"
        directives[3] = f"#SBATCH --error={logs_dir}/{job_name}_%A_%a.err"

    content = '\n'.join(directives) + '\n\n'
    content += 'set -uo pipefail\n\n'
    content += 'START_TIME=$(date +%s)\n\n'
    content += 'echo "=========================================="\n'
    content += f'echo "{job_name.upper()} JOB"\n'
    content += 'echo "=========================================="\n'
    content += 'echo "Job ID: $SLURM_JOB_ID"\n'
    if array:
        content += 'echo "Array Task ID: $SLURM_ARRAY_TASK_ID"\n'
    content += 'echo "Node: $SLURM_NODELIST"\n'
    content += 'echo "Start time: $(date)"\n'
    content += 'echo "=========================================="\n'
    content += 'echo ""\n\n'
    content += "cd $SLURM_SUBMIT_DIR\n\n"
    content += f"# Run command\necho \"Command: {python_cmd}\"\n{python_cmd}\n"
    content += "EXIT_CODE=$?\n\n"
    content += 'END_TIME=$(date +%s)\n'
    content += 'ELAPSED=$((END_TIME - START_TIME))\n'
    content += 'echo ""\n'
    content += 'echo "=========================================="\n'
    content += 'echo "End time: $(date)"\n'
    content += 'printf "Time elapsed: %dh %dm %ds\\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))\n'
    content += 'echo "Exit code: $EXIT_CODE"\n'
    content += 'echo "=========================================="\n\n'
    content += 'exit $EXIT_CODE\n'

    with open(script_path, 'w') as f:
        f.write(content)
    script_path.chmod(0o755)
    return script_path


def submit_job(script_path: Path, dependency: str = None) -> Optional[str]:
    """Submit a Slurm job and return job ID, or None on failure."""
    cmd = ['sbatch']
    if dependency:
        cmd.extend(['--dependency', dependency])
    cmd.append(str(script_path))

    for attempt in range(3):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"  ERROR submitting {script_path.name}: {result.stderr.strip()}")
                return None
            job_id = result.stdout.strip().split()[-1]
            return job_id
        except subprocess.TimeoutExpired:
            print(f"  WARNING: sbatch timed out for {script_path.name} "
                  f"(attempt {attempt+1}/3), retrying in 10s...")
            time.sleep(10)

    print(f"  ERROR: sbatch failed after 3 attempts for {script_path.name}")
    return None


def _collect_job_ids(job_dict) -> List[str]:
    """Extract non-None job IDs from a dict, string, or list (recursive)."""
    if not job_dict:
        return []
    if isinstance(job_dict, str):
        return [job_dict]
    if isinstance(job_dict, dict):
        ids = []
        for v in job_dict.values():
            ids.extend(_collect_job_ids(v))
        return ids
    if isinstance(job_dict, list):
        ids = []
        for v in job_dict:
            ids.extend(_collect_job_ids(v))
        return ids
    return []


def _make_dep_str(job_ids: List[str]) -> Optional[str]:
    """Build afterok dependency string from job IDs, or None if empty."""
    valid = [j for j in job_ids if j is not None]
    if not valid:
        return None
    return f"afterok:{':'.join(valid)}"


def wait_for_jobs(job_ids: List[str], poll_interval: int = 30) -> Dict[str, str]:
    """
    Wait for all Slurm jobs to complete. Returns dict of {job_id: final_state}.

    Uses sacct as primary check (handles completed jobs).
    Detects FAILED/CANCELLED/TIMEOUT and stops waiting for those.
    """
    job_ids = [j for j in job_ids if j]
    if not job_ids:
        return {}

    print(f"\n  Waiting for {len(job_ids)} jobs to complete...")
    # Give Slurm a moment to register the jobs
    time.sleep(5)

    # Terminal states — job is done (successfully or not)
    TERMINAL_STATES = {
        'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT',
        'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED',
        'CANCELLED+', 'DEADLINE',
    }
    # Active states — job is still running or waiting
    ACTIVE_STATES = {
        'RUNNING', 'PENDING', 'CONFIGURING',
        'COMPLETING', 'REQUEUED', 'SUSPENDED',
    }

    final_states = {}
    max_cycles = 2000

    for cycle in range(max_cycles):
        still_active = []

        for jid in job_ids:
            if jid in final_states:
                continue

            base_id = jid.split('_')[0]

            # Primary: use sacct for reliable status
            result = subprocess.run(
                ['sacct', '-j', base_id, '-n', '-X',
                 '-o', 'JobID%30,State%20', '--parsable2'],
                capture_output=True, text=True
            )

            if result.returncode == 0 and result.stdout.strip():
                any_active = False
                any_terminal = False
                terminal_state = None

                for line in result.stdout.strip().split('\n'):
                    parts = line.split('|')
                    if len(parts) < 2:
                        continue
                    state = parts[1].strip()
                    # Handle states like "CANCELLED by 12345"
                    state_base = state.split()[0] if ' ' in state else state

                    if state_base in ACTIVE_STATES:
                        any_active = True
                        still_active.append(f"{parts[0].strip()}({state_base})")
                    elif state_base in TERMINAL_STATES:
                        any_terminal = True
                        terminal_state = state_base

                if any_active:
                    pass  # Still running, keep waiting
                elif any_terminal:
                    final_states[jid] = terminal_state
                    if terminal_state != 'COMPLETED':
                        print(f"  WARNING: Job {jid} ended with state {terminal_state}")
                else:
                    # sacct returned rows but no recognized state — assume done
                    final_states[jid] = 'COMPLETED'
            else:
                # sacct didn't return anything — fall back to squeue
                sq = subprocess.run(
                    ['squeue', '-j', base_id, '-h', '-o', '%A %t'],
                    capture_output=True, text=True
                )
                if sq.returncode == 0 and sq.stdout.strip():
                    still_active.append(f"{jid}(queued)")
                else:
                    # Not in sacct or squeue — assume completed
                    final_states[jid] = 'COMPLETED'

        if not still_active:
            break

        # Print status every ~2 minutes, or first 3 cycles
        if cycle % 4 == 0 or cycle < 3:
            summary = still_active[:5]
            more = f" +{len(still_active)-5} more" if len(still_active) > 5 else ""
            print(f"  {len(still_active)} active: {summary}{more}")

        time.sleep(poll_interval)
    else:
        print(f"  WARNING: Reached max wait cycles. Some jobs may still be running.")

    # Summary
    completed = sum(1 for v in final_states.values() if v == 'COMPLETED')
    failed = {k: v for k, v in final_states.items() if v != 'COMPLETED'}
    if failed:
        print(f"  {completed} completed, {len(failed)} failed: {failed}")
    else:
        print(f"  All {completed} jobs completed successfully!")

    return final_states


# ============================================================================
# PIPELINE STEPS
# ============================================================================

def run_step_preprocess(config, output_dir, scripts_dir, submit, dep_str=None):
    slurm_config = config['slurm']['preprocess_job']
    script_path = create_slurm_script(
        "00_preprocess",
        f"python {scripts_dir}/preprocess.py --config {scripts_dir}/config.yaml",
        "rq2_preprocess", output_dir,
        cpus=slurm_config.get('cpus', 1),
        memory=slurm_config.get('memory', '64G'),
        time_limit=slurm_config.get('time', '00:30:00'),
        partition=config['slurm'].get('partition'),
    )
    print(f"  Script: {script_path.name}")
    if submit:
        job_id = submit_job(script_path, dependency=dep_str)
        print(f"  Submitted: {job_id}")
        return job_id
    return None


def run_step_aggregate(config, output_dir, scripts_dir, submit, dep_str=None):
    tissues = ['subcutaneous', 'visceral', 'white', 'brown']
    slurm_config = config['slurm']['aggregate_job']
    job_ids = {}
    for tissue in tissues:
        script_path = create_slurm_script(
            f"01_aggregate_{tissue}",
            f"python {scripts_dir}/aggregate.py --tissue {tissue} --config {scripts_dir}/config.yaml",
            f"rq2_agg_{tissue[:4]}", output_dir,
            cpus=slurm_config.get('cpus', 1),
            memory=slurm_config.get('memory', '80G'),
            time_limit=slurm_config.get('time', '00:30:00'),
            partition=config['slurm'].get('partition'),
        )
        if submit:
            job_id = submit_job(script_path, dependency=dep_str)
            print(f"  Submitted aggregate ({tissue}): {job_id}")
            job_ids[tissue] = job_id
    return job_ids


def run_step_inference(config, output_dir, scripts_dir, submit,
                       aggregate_jobs=None, dep_str_override=None):
    """Submit inference jobs. Each tissue depends on its own aggregation job."""
    tissues = ['subcutaneous', 'visceral', 'white', 'brown']
    job_ids = {}
    for tissue in tissues:
        for hops in [2, 3]:
            slurm_key = f'psr_{hops}hop_job'
            slurm_config = config['slurm'].get(slurm_key, config['slurm']['psr_2hop_job'])
            script_path = create_slurm_script(
                f"02_psr_{tissue}_{hops}hop",
                f"python {scripts_dir}/run_psr.py --tissue {tissue} --hops {hops} --config {scripts_dir}/config.yaml",
                f"rq2_psr_{tissue[:4]}_{hops}h", output_dir,
                cpus=slurm_config.get('cpus', 4),
                memory=slurm_config.get('memory', '80G'),
                time_limit=slurm_config.get('time', '01:00:00'),
                partition=config['slurm'].get('partition'),
            )
            if submit:
                if dep_str_override is not None:
                    ds = dep_str_override or None  # "" -> None
                elif aggregate_jobs and aggregate_jobs.get(tissue):
                    ds = f"afterok:{aggregate_jobs[tissue]}"
                else:
                    ds = None
                job_id = submit_job(script_path, dependency=ds)
                print(f"  Submitted PSR ({tissue}, {hops}-hop): {job_id}")
                job_ids[(tissue, hops)] = job_id
    return job_ids


def run_step_compare(config, output_dir, scripts_dir, submit,
                     inference_jobs=None, dep_str_override=None):
    comparisons = config['comparisons']
    slurm_config = config['slurm']['compare_job']
    job_ids = {}
    for comp in comparisons:
        name = comp['name']
        tissue_A, tissue_B = comp['tissue_A'], comp['tissue_B']
        script_path = create_slurm_script(
            f"03_compare_{name}",
            f"python {scripts_dir}/compare.py --comparison {name} --config {scripts_dir}/config.yaml",
            f"rq2_cmp_{name[:8]}", output_dir,
            cpus=slurm_config.get('cpus', 1),
            memory=slurm_config.get('memory', '32G'),
            time_limit=slurm_config.get('time', '00:30:00'),
            partition=config['slurm'].get('partition'),
        )
        if submit:
            if dep_str_override is not None:
                ds = dep_str_override or None
            else:
                deps = []
                if inference_jobs:
                    for h in [2, 3]:
                        for t in [tissue_A, tissue_B]:
                            j = inference_jobs.get((t, h))
                            if j:
                                deps.append(j)
                ds = _make_dep_str(deps)
            job_id = submit_job(script_path, dependency=ds)
            print(f"  Submitted compare ({name}): {job_id}")
            job_ids[name] = job_id
    return job_ids


def run_step_enrichment(config, output_dir, scripts_dir, submit,
                        inference_jobs=None, dep_str_override=None):
    comparisons = config['comparisons']
    slurm_config = config['slurm'].get('enrichment_job', {
        'cpus': 1, 'memory': '16G', 'time': '00:15:00'
    })
    job_ids = {}
    for comp in comparisons:
        name = comp['name']
        tissue_A, tissue_B = comp['tissue_A'], comp['tissue_B']
        for hops in [2, 3]:
            script_path = create_slurm_script(
                f"04_enrich_{name}_{hops}hop",
                f"python {scripts_dir}/enrichment.py --comparison {name} --hops {hops} --config {scripts_dir}/config.yaml",
                f"rq2_enr_{name[:6]}_{hops}h", output_dir,
                cpus=slurm_config.get('cpus', 1),
                memory=slurm_config.get('memory', '16G'),
                time_limit=slurm_config.get('time', '00:15:00'),
                partition=config['slurm'].get('partition'),
            )
            if submit:
                if dep_str_override is not None:
                    ds = dep_str_override or None
                else:
                    deps = []
                    if inference_jobs:
                        for t in [tissue_A, tissue_B]:
                            j = inference_jobs.get((t, hops))
                            if j:
                                deps.append(j)
                    ds = _make_dep_str(deps)
                job_id = submit_job(script_path, dependency=ds)
                print(f"  Submitted enrichment ({name}, {hops}-hop): {job_id}")
                job_ids[(name, hops)] = job_id
    return job_ids


def run_step_permutation(config, output_dir, scripts_dir, submit,
                         compare_jobs=None, dep_str_override=None):
    """
    Submit permutation jobs.

    With perms_per_job=100, we need only 100 jobs per comparison.
    Fits in a single Slurm array (0-99) — no batching or chaining needed.
    Each job loads the graph once, runs 100 permutations, saves after each.
    """
    comparisons = config['comparisons']
    slurm_config = config['slurm']['permutation_job']
    n_perms = config['permutation'].get('n_permutations', 1000)
    perms_per_job = config['permutation'].get('perms_per_job', 100)
    max_concurrent_total = config['permutation'].get('max_concurrent_jobs', 10)

    max_concurrent_per_comp = max(1, max_concurrent_total // len(comparisons))

    n_jobs = (n_perms + perms_per_job - 1) // perms_per_job  # ceiling division

    print(f"  Permutations: {n_perms} total, {perms_per_job}/job → "
          f"{n_jobs} jobs per comparison")
    print(f"  Concurrent limit: {max_concurrent_total} global → "
          f"{max_concurrent_per_comp}/comparison")

    job_ids = {}
    for comp in comparisons:
        name = comp['name']

        # perm_start = SLURM_ARRAY_TASK_ID * perms_per_job + 1
        perm_start_expr = f"$((SLURM_ARRAY_TASK_ID * {perms_per_job} + 1))"

        script_path = create_slurm_script(
            f"05_perm_{name}",
            (f"python {scripts_dir}/run_permutation.py "
             f"--perm-start {perm_start_expr} --perm-count {perms_per_job} "
             f"--comparison {name} --config {scripts_dir}/config.yaml"),
            f"rq2_perm_{name[:6]}", output_dir,
            cpus=slurm_config.get('cpus', 1),
            memory=slurm_config.get('memory', '48G'),
            time_limit=slurm_config.get('time', '48:00:00'),
            partition=config['slurm'].get('partition'),
            array=f"0-{n_jobs - 1}%{max_concurrent_per_comp}",
            log_subdir='permutation',
        )

        print(f"  {name}: array 0-{n_jobs-1}, %{max_concurrent_per_comp} concurrent")

        if submit:
            if dep_str_override is not None:
                ds = dep_str_override or None
            elif compare_jobs and compare_jobs.get(name):
                ds = f"afterok:{compare_jobs[name]}"
            else:
                ds = None

            job_id = submit_job(script_path, dependency=ds)
            print(f"    Submitted: {job_id}")
            job_ids[name] = job_id
            time.sleep(1)

    return job_ids


def run_step_permutation_aggregate(config, output_dir, scripts_dir, submit,
                                    perm_jobs=None, dep_str_override=None):
    comparisons = config['comparisons']
    slurm_config = config['slurm'].get('permutation_aggregate_job', {
        'cpus': 1, 'memory': '16G', 'time': '00:15:00'
    })
    job_ids = {}
    for comp in comparisons:
        name = comp['name']
        script_path = create_slurm_script(
            f"06_perm_agg_{name}",
            f"python {scripts_dir}/aggregate_permutations.py --comparison {name} --config {scripts_dir}/config.yaml",
            f"rq2_pagg_{name[:6]}", output_dir,
            cpus=slurm_config.get('cpus', 1),
            memory=slurm_config.get('memory', '16G'),
            time_limit=slurm_config.get('time', '00:15:00'),
            partition=config['slurm'].get('partition'),
        )
        if submit:
            if dep_str_override is not None:
                ds = dep_str_override or None
            elif perm_jobs and perm_jobs.get(name):
                # Batches are chained — only the LAST job ID matters
                # (if it completed, all prior batches completed too)
                perm_ids = perm_jobs[name]
                if isinstance(perm_ids, list) and perm_ids:
                    ds = f"afterok:{perm_ids[-1]}"
                elif isinstance(perm_ids, str):
                    ds = f"afterok:{perm_ids}"
                else:
                    ds = None
            else:
                ds = None
            job_id = submit_job(script_path, dependency=ds)
            print(f"  Submitted perm aggregate ({name}): {job_id}")
            job_ids[name] = job_id
    return job_ids


def run_step_report(config, output_dir, scripts_dir, submit,
                    compare_deps=None, enrich_deps=None, perm_deps=None,
                    dep_str_override=None):
    comparisons = config['comparisons']
    slurm_config = config['slurm']['report_job']
    job_ids = {}
    for comp in comparisons:
        name = comp['name']
        for hops in [2, 3]:
            script_path = create_slurm_script(
                f"07_report_{name}_{hops}hop",
                f"python {scripts_dir}/generate_report.py --comparison {name} --hops {hops} --config {scripts_dir}/config.yaml",
                f"rq2_rpt_{name[:6]}_{hops}h", output_dir,
                cpus=slurm_config.get('cpus', 1),
                memory=slurm_config.get('memory', '20G'),
                time_limit=slurm_config.get('time', '00:30:00'),
                partition=config['slurm'].get('partition'),
            )
            if submit:
                if dep_str_override is not None:
                    ds = dep_str_override or None
                else:
                    deps = []
                    if compare_deps and compare_deps.get(name):
                        deps.append(compare_deps[name])
                    if enrich_deps and enrich_deps.get((name, hops)):
                        deps.append(enrich_deps[(name, hops)])
                    if perm_deps and perm_deps.get(name):
                        deps.append(perm_deps[name])
                    ds = _make_dep_str(deps)
                job_id = submit_job(script_path, dependency=ds)
                print(f"  Submitted report ({name}, {hops}-hop): {job_id}")
                job_ids[(name, hops)] = job_id
    return job_ids


def run_step_volcano(config, output_dir, scripts_dir, submit,
                     perm_agg_deps=None, dep_str_override=None):
    """Step 8: Volcano plots (depends on perm_aggregate for p/q columns)."""
    comparisons = config['comparisons']
    slurm_config = config['slurm'].get('visualization_job', {
        'cpus': 1, 'memory': '16G', 'time': '00:30:00'
    })
    job_ids = {}
    for comp in comparisons:
        name = comp['name']
        for hops in [2, 3]:
            script_path = create_slurm_script(
                f"08_volcano_{name}_{hops}hop",
                f"python {scripts_dir}/plot_volcano.py --comparison {name} --hops {hops} --config {scripts_dir}/config.yaml",
                f"rq2_vol_{name[:6]}_{hops}h", output_dir,
                cpus=slurm_config.get('cpus', 1),
                memory=slurm_config.get('memory', '16G'),
                time_limit=slurm_config.get('time', '00:30:00'),
                partition=config['slurm'].get('partition'),
            )
            if submit:
                if dep_str_override is not None:
                    ds = dep_str_override or None
                else:
                    deps = []
                    if perm_agg_deps and perm_agg_deps.get(name):
                        deps.append(perm_agg_deps[name])
                    ds = _make_dep_str(deps)
                job_id = submit_job(script_path, dependency=ds)
                print(f"  Submitted volcano ({name}, {hops}-hop): {job_id}")
                job_ids[(name, hops)] = job_id
    return job_ids


def run_step_visualizations(config, output_dir, scripts_dir, submit,
                            perm_agg_deps=None, dep_str_override=None):
    """Step 9: Result visualizations (depends on perm_aggregate for p/q cols)."""
    comparisons = config['comparisons']
    slurm_config = config['slurm'].get('visualization_job', {
        'cpus': 1, 'memory': '16G', 'time': '00:30:00'
    })
    job_ids = {}
    for comp in comparisons:
        name = comp['name']
        script_path = create_slurm_script(
            f"09_plots_{name}",
            f"python {scripts_dir}/plot_results.py --comparison {name} --config {scripts_dir}/config.yaml",
            f"rq2_plt_{name[:6]}", output_dir,
            cpus=slurm_config.get('cpus', 1),
            memory=slurm_config.get('memory', '16G'),
            time_limit=slurm_config.get('time', '00:30:00'),
            partition=config['slurm'].get('partition'),
        )
        if submit:
            if dep_str_override is not None:
                ds = dep_str_override or None
            else:
                deps = []
                if perm_agg_deps and perm_agg_deps.get(name):
                    deps.append(perm_agg_deps[name])
                ds = _make_dep_str(deps)
            job_id = submit_job(script_path, dependency=ds)
            print(f"  Submitted visualizations ({name}): {job_id}")
            job_ids[name] = job_id
    return job_ids


# ============================================================================
# FULL PIPELINE
# ============================================================================

def _wait_and_check(job_dict, step_name: str) -> bool:
    """Wait for jobs, check for failures. Returns True if all OK."""
    ids = _collect_job_ids(job_dict)
    if not ids:
        print(f"  No {step_name} jobs to wait for")
        return True

    states = wait_for_jobs(ids)
    failed = {k: v for k, v in states.items() if v != 'COMPLETED'}
    if failed:
        print(f"\n  WARNING: {len(failed)} {step_name} jobs failed.")
        print(f"  Pipeline continues but downstream steps may fail.")
        return False
    return True


def run_full_pipeline(config: dict, submit: bool = False, wait: bool = False):
    """
    Run the complete RQ2 pipeline.

    When wait=True:  submit each step WITHOUT Slurm dependencies, wait for
                     completion, then submit the next step. This avoids the
                     problem of afterok on already-completed jobs.
    When wait=False: submit all steps with Slurm afterok dependency chains.
    """
    output_dir = Path(config['paths']['output_dir'])
    scripts_dir = Path(config['paths']['scripts_dir'])

    print("=" * 60)
    print("RQ2 PIPELINE ORCHESTRATOR")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Scripts: {scripts_dir}")
    mode = ('submit + wait (sequential)' if wait
            else 'submit with dependency chains' if submit
            else 'dry run (scripts only)')
    print(f"Mode: {mode}")
    print()

    # When wait=True we pass dep_str_override="" to all steps, which becomes
    # None inside the functions (no Slurm dependencies). The orchestrator
    # itself serializes steps by waiting.
    NO_DEP = ""  # Sentinel: means "I've already waited, no dep needed"

    all_jobs = {}

    # Step 0: Preprocess
    print("\n--- Step 0: Preprocess ---")
    preprocess_job = run_step_preprocess(config, output_dir, scripts_dir, submit)
    all_jobs['preprocess'] = preprocess_job
    if wait and submit:
        _wait_and_check(preprocess_job, 'preprocess')

    # Step 1: Aggregate
    print("\n--- Step 1: Aggregate (4 tissues) ---")
    if wait:
        agg_jobs = run_step_aggregate(config, output_dir, scripts_dir, submit)
    else:
        dep = f"afterok:{preprocess_job}" if preprocess_job else None
        agg_jobs = run_step_aggregate(config, output_dir, scripts_dir, submit, dep_str=dep)
    all_jobs['aggregate'] = agg_jobs
    if wait and submit:
        _wait_and_check(agg_jobs, 'aggregate')

    # Step 2: Inference
    print("\n--- Step 2: Inference (4 tissues × 2 hops) ---")
    if wait:
        inf_jobs = run_step_inference(config, output_dir, scripts_dir, submit,
                                      dep_str_override=NO_DEP)
    else:
        inf_jobs = run_step_inference(config, output_dir, scripts_dir, submit,
                                      aggregate_jobs=agg_jobs)
    all_jobs['inference'] = inf_jobs
    if wait and submit:
        _wait_and_check(inf_jobs, 'inference')

    # Step 3: Compare
    print("\n--- Step 3: Compare (2 comparisons) ---")
    if wait:
        cmp_jobs = run_step_compare(config, output_dir, scripts_dir, submit,
                                     dep_str_override=NO_DEP)
    else:
        cmp_jobs = run_step_compare(config, output_dir, scripts_dir, submit,
                                     inference_jobs=inf_jobs)
    all_jobs['compare'] = cmp_jobs

    # Step 4: Enrichment (depends on inference, not compare — runs in parallel with compare)
    print("\n--- Step 4: Enrichment (2 comparisons × 2 hops) ---")
    if wait:
        enr_jobs = run_step_enrichment(config, output_dir, scripts_dir, submit,
                                        dep_str_override=NO_DEP)
    else:
        enr_jobs = run_step_enrichment(config, output_dir, scripts_dir, submit,
                                        inference_jobs=inf_jobs)
    all_jobs['enrichment'] = enr_jobs

    # Wait for both compare and enrichment
    if wait and submit:
        _wait_and_check(cmp_jobs, 'compare')
        _wait_and_check(enr_jobs, 'enrichment')

    # Step 5: Permutation
    print("\n--- Step 5: Permutation Testing ---")
    if wait:
        perm_jobs = run_step_permutation(config, output_dir, scripts_dir, submit,
                                          dep_str_override=NO_DEP)
    else:
        perm_jobs = run_step_permutation(config, output_dir, scripts_dir, submit,
                                          compare_jobs=cmp_jobs)
    all_jobs['permutation'] = perm_jobs
    if wait and submit:
        _wait_and_check(perm_jobs, 'permutation')

    # Step 6: Aggregate Permutations → compute per-triple p-values
    print("\n--- Step 6: Compute Per-Triple P-values (aggregate permutations) ---")
    if wait:
        pagg_jobs = run_step_permutation_aggregate(config, output_dir, scripts_dir, submit,
                                                    dep_str_override=NO_DEP)
    else:
        pagg_jobs = run_step_permutation_aggregate(config, output_dir, scripts_dir, submit,
                                                    perm_jobs=perm_jobs)
    all_jobs['perm_aggregate'] = pagg_jobs
    if wait and submit:
        _wait_and_check(pagg_jobs, 'perm_aggregate')

    # Step 7: Reports (depends on compare + enrichment + perm p-values)
    print("\n--- Step 7: Generate Reports ---")
    if wait:
        rpt_jobs = run_step_report(config, output_dir, scripts_dir, submit,
                                    dep_str_override=NO_DEP)
    else:
        rpt_jobs = run_step_report(config, output_dir, scripts_dir, submit,
                                    compare_deps=cmp_jobs, enrich_deps=enr_jobs,
                                    perm_deps=pagg_jobs)
    all_jobs['report'] = rpt_jobs
    if wait and submit:
        _wait_and_check(rpt_jobs, 'report')

    # Step 8: Volcano plots (depends on perm p-values in comparison parquets)
    print("\n--- Step 8: Volcano Plots ---")
    if wait:
        vol_jobs = run_step_volcano(config, output_dir, scripts_dir, submit,
                                     dep_str_override=NO_DEP)
    else:
        vol_jobs = run_step_volcano(config, output_dir, scripts_dir, submit,
                                     perm_agg_deps=pagg_jobs)
    all_jobs['volcano'] = vol_jobs
    if wait and submit:
        _wait_and_check(vol_jobs, 'volcano')

    # Step 9: Result visualizations (depends on perm p-values)
    print("\n--- Step 9: Result Visualizations ---")
    if wait:
        viz_jobs = run_step_visualizations(config, output_dir, scripts_dir, submit,
                                            dep_str_override=NO_DEP)
    else:
        viz_jobs = run_step_visualizations(config, output_dir, scripts_dir, submit,
                                            perm_agg_deps=pagg_jobs)
    all_jobs['visualizations'] = viz_jobs
    if wait and submit:
        _wait_and_check(viz_jobs, 'visualizations')

    print("\n" + "=" * 60)
    status = ("PIPELINE COMPLETE" if wait
              else "PIPELINE SUBMITTED" if submit
              else "DRY RUN COMPLETE")
    print(status)
    print("=" * 60)

    if submit:
        for step, jobs in all_jobs.items():
            if isinstance(jobs, dict):
                n = sum(1 for v in jobs.values() if v is not None)
                failed = sum(1 for v in jobs.values() if v is None)
                extra = f" ({failed} failed to submit)" if failed else ""
                print(f"  {step}: {n} jobs{extra}")
            elif jobs:
                print(f"  {step}: {jobs}")
            else:
                print(f"  {step}: not submitted")

    return all_jobs


def main():
    parser = argparse.ArgumentParser(description='RQ2 Pipeline Orchestrator')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--submit', action='store_true', help='Submit jobs to Slurm')
    parser.add_argument('--wait', action='store_true', help='Wait for each step to complete')
    parser.add_argument('--step', type=str, choices=[
        'preprocess', 'aggregate', 'inference', 'compare', 'enrichment',
        'permutation', 'perm_aggregate', 'report', 'volcano', 'visualizations', 'all'
    ], default='all', help='Run specific step only (no dependencies)')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.step == 'all':
        run_full_pipeline(config, args.submit, args.wait)
    else:
        output_dir = Path(config['paths']['output_dir'])
        scripts_dir = Path(config['paths']['scripts_dir'])

        step_funcs = {
            'preprocess': lambda: run_step_preprocess(config, output_dir, scripts_dir, args.submit),
            'aggregate': lambda: run_step_aggregate(config, output_dir, scripts_dir, args.submit),
            'inference': lambda: run_step_inference(config, output_dir, scripts_dir, args.submit),
            'compare': lambda: run_step_compare(config, output_dir, scripts_dir, args.submit),
            'enrichment': lambda: run_step_enrichment(config, output_dir, scripts_dir, args.submit),
            'permutation': lambda: run_step_permutation(config, output_dir, scripts_dir, args.submit),
            'perm_aggregate': lambda: run_step_permutation_aggregate(config, output_dir, scripts_dir, args.submit),
            'report': lambda: run_step_report(config, output_dir, scripts_dir, args.submit),
            'volcano': lambda: run_step_volcano(config, output_dir, scripts_dir, args.submit),
            'visualizations': lambda: run_step_visualizations(config, output_dir, scripts_dir, args.submit),
        }

        jobs = step_funcs[args.step]()

        if args.wait and args.submit:
            ids = _collect_job_ids(jobs)
            if ids:
                wait_for_jobs(ids)


if __name__ == '__main__':
    main()