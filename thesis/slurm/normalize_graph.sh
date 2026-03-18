#!/bin/bash
#SBATCH --job-name=normalize_graph
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail

echo "=========================================="
echo "NORMALIZE GRAPH FIELDS"
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


python /home/projects2/ContextAwareKGReasoning/ContextAwareKGReasoning/scripts/normalize_graph.py \
    /home/projects2/ContextAwareKGReasoning/data/graphs/augmented_graph_annotated_combined.pkl \
    /home/projects2/ContextAwareKGReasoning/data/graphs/augmented_graph_normalized.pkl

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