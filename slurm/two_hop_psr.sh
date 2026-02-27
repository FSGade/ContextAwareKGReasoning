#!/bin/bash
#SBATCH --job-name=psr_2hop
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --export=ALL,PYTHONUNBUFFERED=1
set -euo pipefail

echo "=========================================="
echo "PSR TWO-HOP INFERENCE"
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


cd /home/projects2/ContextAwareKGReasoning/ContextAwareKGReasoning/

python /home/projects2/ContextAwareKGReasoning/ContextAwareKGReasoning/scripts/psr/psr.py

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