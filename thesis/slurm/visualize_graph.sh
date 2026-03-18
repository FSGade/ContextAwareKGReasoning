#!/bin/bash
#SBATCH --job-name=kg_viz
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --export=ALL,PYTHONUNBUFFERED=1
set -euo pipefail

echo "=========================================="
echo "KNOWLEDGE GRAPH VISUALIZATION"
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

export PYTHONPATH="/home/projects2/ContextAwareKGReasoning:$PYTHONPATH"
START_TIME=$(date +%s)


GRAPH_INPUT="/home/projects2/ContextAwareKGReasoning/data/graphs/ikraph.pkl"
OUTPUT_DIR="/home/projects2/ContextAwareKGReasoning/results/visualizations/ikraph_full"

mkdir -p "$OUTPUT_DIR"

python /home/projects2/ContextAwareKGReasoning/ContextAwareKGReasoning/scripts/viz_graph.py \
    --input "$GRAPH_INPUT" \
    --output "$OUTPUT_DIR"

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