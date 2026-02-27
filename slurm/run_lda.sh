#!/bin/bash
#SBATCH --job-name=lda_fit
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail

echo "=========================================="
echo "LDA TOPIC MODELING"
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


# Toggle mode: "explore" or "fit"
MODE="fit"

INPUT_GRAPH="/home/projects2/ContextAwareKGReasoning/data/graphs/augmented_graph_normalized.pkl"
OUTPUT_DIR="/home/projects2/ContextAwareKGReasoning/results/lda_output_fit_final"
OUTPUT_GRAPH="/home/projects2/ContextAwareKGReasoning/data/graphs/augmented_graph_with_topics.pkl"

K_MECHANISMS=8
K_PATHWAYS=9
MIN_DF=2
MAX_DF_FRAC=0.50

echo "Mode: $MODE"
echo "Input: $INPUT_GRAPH"
echo ""

if [ "$MODE" = "explore" ]; then
    python /home/projects2/ContextAwareKGReasoning/ContextAwareKGReasoning/scripts/fit_lda.py \
        --mode explore \
        --input "$INPUT_GRAPH" \
        --output-dir "$OUTPUT_DIR" \
        --min-df $MIN_DF --max-df-frac $MAX_DF_FRAC

elif [ "$MODE" = "fit" ]; then
    python /home/projects2/ContextAwareKGReasoning/ContextAwareKGReasoning/scripts/fit_lda.py \
        --mode fit \
        --input "$INPUT_GRAPH" \
        --output "$OUTPUT_GRAPH" \
        --output-dir "$OUTPUT_DIR" \
        --k-mechanisms $K_MECHANISMS --k-pathways $K_PATHWAYS \
        --min-df $MIN_DF --max-df-frac $MAX_DF_FRAC
else
    echo "ERROR: MODE must be 'explore' or 'fit'"
    exit 1
fi

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