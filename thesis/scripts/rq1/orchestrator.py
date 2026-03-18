#!/usr/bin/env python3
"""
RQ1 Master Orchestrator: Manages the complete tissue context analysis pipeline.

This script:
1. Creates Slurm job scripts for each step
2. Submits jobs with proper dependencies
3. Monitors job completion
4. Handles failures and retries

Usage:
    # Dry run (print commands without submitting)
    python orchestrator.py --config config.yaml
    
    # Actually submit jobs
    python orchestrator.py --config config.yaml --submit
    
    # Run specific step only
    python orchestrator.py --config config.yaml --submit --step filter
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import yaml


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_slurm_script(
    script_name: str,
    python_cmd: str,
    job_name: str,
    output_dir: Path,
    cpus: int = 1,
    memory: str = "32G",
    time_limit: str = "01:00:00",
    partition: str = None,
) -> Path:
    """
    Create a Slurm job script.
    
    Returns path to the created script.
    """
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = output_dir / "slurm_scripts" / f"{script_name}.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    partition_line = f"#SBATCH --partition={partition}" if partition else ""
    
    script_content = dedent(f'''
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --output={logs_dir}/{job_name}_%j.out
        #SBATCH --error={logs_dir}/{job_name}_%j.err
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --mem={memory}
        #SBATCH --time={time_limit}
        {partition_line}
        
        # Print job info
        echo "========================================"
        echo "Job ID: $SLURM_JOB_ID"
        echo "Job name: {job_name}"
        echo "Node: $(hostname)"
        echo "CPUs: {cpus}"
        echo "Memory: {memory}"
        echo "Start time: $(date)"
        echo "========================================"
        echo ""
        
        # Run the Python command
        {python_cmd}
        
        EXIT_CODE=$?
        
        echo ""
        echo "========================================"
        echo "End time: $(date)"
        echo "Exit code: $EXIT_CODE"
        echo "========================================"
        
        exit $EXIT_CODE
    ''').strip()
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path


def submit_job(script_path: Path, dependency: str = None) -> str:
    """
    Submit a Slurm job and return the job ID.
    
    Args:
        script_path: Path to the Slurm script
        dependency: Dependency string (e.g., "afterok:12345")
    
    Returns:
        Job ID as string
    """
    cmd = ['sbatch', '--parsable']
    if dependency:
        cmd.append(f'--dependency={dependency}')
    cmd.append(str(script_path))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit job: {result.stderr}")
    
    job_id = result.stdout.strip()
    return job_id


def wait_for_job(job_id: str, poll_interval: int = 30, timeout: int = 86400) -> str:
    """
    Wait for a Slurm job to complete.
    
    Args:
        job_id: Slurm job ID
        poll_interval: Seconds between status checks
        timeout: Maximum seconds to wait
    
    Returns:
        Final job status
    """
    start_time = time.time()
    
    while True:
        # Check if job is still in queue
        result = subprocess.run(
            ['squeue', '-j', job_id, '-h', '-o', '%T'],
            capture_output=True, text=True
        )
        
        status = result.stdout.strip()
        
        if not status:
            # Job no longer in queue - check final status
            sacct_result = subprocess.run(
                ['sacct', '-j', job_id, '--format=State', '--noheader', '-P'],
                capture_output=True, text=True
            )
            final_status = sacct_result.stdout.strip().split('\n')[0]
            return final_status
        
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")
        
        print(f"  Job {job_id}: {status} (elapsed: {int(elapsed)}s)")
        time.sleep(poll_interval)


def wait_for_jobs(job_ids: list, poll_interval: int = 60) -> dict:
    """
    Wait for multiple jobs to complete.
    
    Returns dict mapping job_id -> final_status
    """
    results = {}
    pending = set(job_ids)
    
    while pending:
        completed = set()
        
        for job_id in pending:
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h'],
                capture_output=True, text=True
            )
            
            if not result.stdout.strip():
                # Job completed
                sacct_result = subprocess.run(
                    ['sacct', '-j', job_id, '--format=State', '--noheader', '-P'],
                    capture_output=True, text=True
                )
                final_status = sacct_result.stdout.strip().split('\n')[0]
                results[job_id] = final_status
                completed.add(job_id)
        
        pending -= completed
        
        if pending:
            print(f"  Waiting for {len(pending)} jobs: {list(pending)}")
            time.sleep(poll_interval)
    
    return results


def run_step_filter(config: dict, output_dir: Path, scripts_dir: Path, submit: bool) -> str:
    """Create and optionally submit the filter job."""
    slurm_config = config['slurm']['filter_job']
    
    python_cmd = f"python {scripts_dir}/filter_graphs.py --config {scripts_dir}/config.yaml"
    
    script_path = create_slurm_script(
        script_name="01_filter",
        python_cmd=python_cmd,
        job_name="rq1_filter",
        output_dir=output_dir,
        cpus=slurm_config.get('cpus', 1),
        memory=slurm_config.get('memory', '80G'),
        time_limit=slurm_config.get('time', '00:30:00'),
        partition=config['slurm'].get('partition'),
    )
    
    print(f"Created filter script: {script_path}")
    
    if submit:
        job_id = submit_job(script_path)
        print(f"Submitted filter job: {job_id}")
        return job_id
    
    return None


def run_step_aggregate(config: dict, output_dir: Path, scripts_dir: Path, submit: bool,
                       dependency: str = None) -> list:
    """Create and optionally submit aggregate jobs for all contexts."""
    contexts = ['baseline', 'adipose', 'nonadipose', 'liver']
    slurm_config = config['slurm']['aggregate_job']
    
    job_ids = []
    
    for ctx in contexts:
        python_cmd = (
            f"python {scripts_dir}/aggregate_graphs.py "
            f"--context {ctx} --config {scripts_dir}/config.yaml"
        )
        
        script_path = create_slurm_script(
            script_name=f"01b_aggregate_{ctx}",
            python_cmd=python_cmd,
            job_name=f"rq1_agg_{ctx}",
            output_dir=output_dir,
            cpus=slurm_config.get('cpus', 1),
            memory=slurm_config.get('memory', '100G'),
            time_limit=slurm_config.get('time', '01:00:00'),
            partition=config['slurm'].get('partition'),
        )
        
        print(f"Created aggregate script: {script_path}")
        
        if submit:
            dep_str = f"afterok:{dependency}" if dependency else None
            job_id = submit_job(script_path, dependency=dep_str)
            print(f"Submitted aggregate job ({ctx}): {job_id}")
            job_ids.append(job_id)
    
    return job_ids


def run_step_psr(config: dict, output_dir: Path, scripts_dir: Path, submit: bool, 
                 dependency: str = None) -> list:
    """Create and optionally submit PSR jobs for all contexts and hop lengths."""
    contexts = ['baseline', 'adipose', 'nonadipose', 'liver']
    hops = [2, 3]
    
    job_ids = []
    
    for hop in hops:
        slurm_key = f'psr_{hop}hop_job'
        slurm_config = config['slurm'].get(slurm_key, config['slurm']['psr_2hop_job'])
        
        for ctx in contexts:
            python_cmd = (
                f"python {scripts_dir}/run_psr.py "
                f"--context {ctx} --hops {hop} --config {scripts_dir}/config.yaml"
            )
            
            script_path = create_slurm_script(
                script_name=f"02_psr_{ctx}_{hop}hop",
                python_cmd=python_cmd,
                job_name=f"rq1_psr_{ctx}_{hop}h",
                output_dir=output_dir,
                cpus=slurm_config.get('cpus', 4),
                memory=slurm_config.get('memory', '80G'),
                time_limit=slurm_config.get('time', '02:00:00'),
                partition=config['slurm'].get('partition'),
            )
            
            print(f"Created PSR script: {script_path}")
            
            if submit:
                dep_str = f"afterok:{dependency}" if dependency else None
                job_id = submit_job(script_path, dependency=dep_str)
                print(f"Submitted PSR job ({ctx}, {hop}-hop): {job_id}")
                job_ids.append(job_id)
    
    return job_ids


def run_step_compare(config: dict, output_dir: Path, scripts_dir: Path, submit: bool,
                     dependency: str = None) -> str:
    """Create and optionally submit the comparison job."""
    slurm_config = config['slurm']['compare_job']
    
    python_cmd = f"python {scripts_dir}/compare_contexts.py --config {scripts_dir}/config.yaml"
    
    script_path = create_slurm_script(
        script_name="03_compare",
        python_cmd=python_cmd,
        job_name="rq1_compare",
        output_dir=output_dir,
        cpus=slurm_config.get('cpus', 1),
        memory=slurm_config.get('memory', '16G'),
        time_limit=slurm_config.get('time', '00:30:00'),
        partition=config['slurm'].get('partition'),
    )
    
    print(f"Created compare script: {script_path}")
    
    if submit:
        dep_str = f"afterok:{dependency}" if dependency else None
        job_id = submit_job(script_path, dependency=dep_str)
        print(f"Submitted compare job: {job_id}")
        return job_id
    
    return None


def run_step_report(config: dict, output_dir: Path, scripts_dir: Path, submit: bool,
                    dependency: str = None) -> str:
    """Create and optionally submit the report generation job."""
    slurm_config = config['slurm']['report_job']
    
    python_cmd = f"python {scripts_dir}/generate_report.py --config {scripts_dir}/config.yaml"
    
    script_path = create_slurm_script(
        script_name="04_report",
        python_cmd=python_cmd,
        job_name="rq1_report",
        output_dir=output_dir,
        cpus=slurm_config.get('cpus', 1),
        memory=slurm_config.get('memory', '8G'),
        time_limit=slurm_config.get('time', '00:30:00'),
        partition=config['slurm'].get('partition'),
    )
    
    print(f"Created report script: {script_path}")
    
    if submit:
        dep_str = f"afterok:{dependency}" if dependency else None
        job_id = submit_job(script_path, dependency=dep_str)
        print(f"Submitted report job: {job_id}")
        return job_id
    
    return None


def main():
    parser = argparse.ArgumentParser(description='RQ1 Pipeline Orchestrator')
    parser.add_argument('--config', type=Path, required=True, help='Path to config.yaml')
    parser.add_argument('--submit', action='store_true', help='Actually submit jobs (default: dry run)')
    parser.add_argument('--step', choices=['filter', 'aggregate', 'psr', 'compare', 'report', 'all'],
                        default='all', help='Run specific step only')
    parser.add_argument('--wait', action='store_true', help='Wait for jobs to complete')
    parser.add_argument('--dependency', type=str, help='Job ID to depend on (for partial runs)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Paths
    output_dir = Path(config['paths']['output_dir'])
    scripts_dir = args.config.parent
    
    print("=" * 80)
    print("RQ1 PIPELINE ORCHESTRATOR")
    print("=" * 80)
    print(f"\nConfig: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Scripts directory: {scripts_dir}")
    print(f"Mode: {'SUBMIT' if args.submit else 'DRY RUN'}")
    print(f"Step: {args.step}")
    
    if not args.submit:
        print("\n[WARN] DRY RUN MODE - Jobs will not be submitted. Use --submit to submit jobs.")
    
    # Track job IDs
    all_jobs = {}
    
    # Step 1: Filter
    if args.step in ['all', 'filter']:
        print("\n" + "-" * 40)
        print("STEP 1: FILTER GRAPHS")
        print("-" * 40)
        
        filter_job_id = run_step_filter(config, output_dir, scripts_dir, args.submit)
        if filter_job_id:
            all_jobs['filter'] = filter_job_id
    
    # Step 1b: Aggregate
    if args.step in ['all', 'aggregate']:
        print("\n" + "-" * 40)
        print("STEP 1b: AGGREGATE GRAPHS")
        print("-" * 40)
        
        # Determine dependency
        if args.step == 'all' and 'filter' in all_jobs:
            agg_dependency = all_jobs['filter']
        elif args.dependency:
            agg_dependency = args.dependency
        else:
            agg_dependency = None
        
        agg_job_ids = run_step_aggregate(config, output_dir, scripts_dir, args.submit, agg_dependency)
        if agg_job_ids:
            all_jobs['aggregate'] = agg_job_ids
    
    # Step 2: PSR
    if args.step in ['all', 'psr']:
        print("\n" + "-" * 40)
        print("STEP 2: PSR INFERENCE")
        print("-" * 40)
        
        # Determine dependency (all aggregate jobs must complete)
        if args.step == 'all' and 'aggregate' in all_jobs:
            psr_dependency = ':'.join(all_jobs['aggregate'])
        elif args.dependency:
            psr_dependency = args.dependency
        else:
            psr_dependency = None
        
        psr_job_ids = run_step_psr(config, output_dir, scripts_dir, args.submit, psr_dependency)
        if psr_job_ids:
            all_jobs['psr'] = psr_job_ids
    
    # Step 3: Compare
    if args.step in ['all', 'compare']:
        print("\n" + "-" * 40)
        print("STEP 3: COMPARE CONTEXTS")
        print("-" * 40)
        
        # Determine dependency (all PSR jobs must complete)
        if args.step == 'all' and 'psr' in all_jobs:
            compare_dependency = ':'.join(all_jobs['psr'])
        elif args.dependency:
            compare_dependency = args.dependency
        else:
            compare_dependency = None
        
        compare_job_id = run_step_compare(config, output_dir, scripts_dir, args.submit, compare_dependency)
        if compare_job_id:
            all_jobs['compare'] = compare_job_id
    
    # Step 4: Report
    if args.step in ['all', 'report']:
        print("\n" + "-" * 40)
        print("STEP 4: GENERATE REPORTS")
        print("-" * 40)
        
        # Determine dependency
        if args.step == 'all' and 'compare' in all_jobs:
            report_dependency = all_jobs['compare']
        elif args.dependency:
            report_dependency = args.dependency
        else:
            report_dependency = None
        
        report_job_id = run_step_report(config, output_dir, scripts_dir, args.submit, report_dependency)
        if report_job_id:
            all_jobs['report'] = report_job_id
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if args.submit:
        print("\nSubmitted jobs:")
        for step, job_ids in all_jobs.items():
            if isinstance(job_ids, list):
                print(f"  {step}: {', '.join(job_ids)}")
            else:
                print(f"  {step}: {job_ids}")
        
        print(f"\nMonitor with: squeue -u $USER")
        print(f"Logs in: {output_dir}/logs/")
        
        # Wait for jobs if requested
        if args.wait and all_jobs:
            print("\nWaiting for jobs to complete...")
            
            # Flatten job IDs
            all_job_ids = []
            for job_ids in all_jobs.values():
                if isinstance(job_ids, list):
                    all_job_ids.extend(job_ids)
                else:
                    all_job_ids.append(job_ids)
            
            results = wait_for_jobs(all_job_ids)
            
            print("\nJob results:")
            for job_id, status in results.items():
                status_icon = "OK" if status == "COMPLETED" else "FAIL"
                print(f"  {status_icon} {job_id}: {status}")
            
            # Check for failures
            failures = [jid for jid, status in results.items() if status != "COMPLETED"]
            if failures:
                print(f"\n[WARN] {len(failures)} job(s) failed!")
                sys.exit(1)
            else:
                print("\nOK All jobs completed successfully!")
    else:
        print("\nSlurm scripts created in: {}/slurm_scripts/".format(output_dir))
        print("\nTo submit jobs, run with --submit flag:")
        print(f"  python {sys.argv[0]} --config {args.config} --submit")
    
    print(f"\nResults will be saved to: {output_dir}")


if __name__ == '__main__':
    main()