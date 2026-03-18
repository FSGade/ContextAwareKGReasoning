#!/bin/bash
#SBATCH --job-name=sub_aug
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=15:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail

echo "=========================================="
echo "SEARCH, SUBSET AND AUGMENT"
echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="
echo ""

mkdir -p logs

# Activate virtual environment
source /home/projects2/ContextAwareKGReasoning/kgenv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
START_TIME=$(date +%s)


BASE_GRAPH="/home/projects2/ContextAwareKGReasoning/data/graphs/subsets/ikraph_pubmed_human.pkl"
BASE_OUTPUT="/home/projects2/ContextAwareKGReasoning/data/kg_subset/run_$(date +%Y%m%d)"
SCRIPT="/home/projects2/ContextAwareKGReasoning/ContextAwareKGReasoning/scripts/subset_and_augment.py"
SEED=42
STRATEGIES=("greedy" "random" "weighted")

mkdir -p "${BASE_OUTPUT}"

for STRATEGY in "${STRATEGIES[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT}/${STRATEGY}"
    echo ""
    echo "Running strategy: ${STRATEGY} -> ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"

    python "${SCRIPT}" \
        --base-graph "${BASE_GRAPH}" \
        --output-dir "${OUTPUT_DIR}" \
        --email "your.email@dtu.dk" \
        --start-year 2000 --end-year 2023 \
        --max-edges 100000 \
        --sampling-strategy "${STRATEGY}" \
        --only-directed --random-seed "${SEED}"

    if [ $? -ne 0 ]; then
        echo "ERROR: Strategy ${STRATEGY} failed"
        break
    fi
done

EXIT_STATUS=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "End time:      $(date)"
echo "Time elapsed:  ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Exit status:   $EXIT_STATUS"
echo "=========================================="

exit $EXIT_STATUS