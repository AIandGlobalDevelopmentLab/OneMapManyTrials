#!/usr/bin/env bash
#SBATCH -A naiss2024-5-450 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 0-02:00:00
#SBATCH --array=0-23     # Adjust based on number of lines in train_models_array_args.txt
#SBATCH -o ../logs/slurm-%A/%a.out
#SBATCH -e ../logs/slurm-%A/%a.err

module purge

# Load config.ini
source <(grep = ../config.ini)

echo "=== CONFIG VALUES ==="
echo "DATA_DIR: $DATA_DIR";
echo "APPTAINER_IMG: $APPTAINER_IMG"
echo

# Read arguments for this job index
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" train_models_array_args.txt)

echo "Running with args: $ARGS"

apptainer exec $APPTAINER_IMG python train_model.py $ARGS
