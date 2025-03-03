#!/usr/bin/env bash
#SBATCH -A naiss2024-5-450 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1  # We’re launching 1 node with 1 Nvidia A40 GPU
#SBATCH --job-name=ate_analysis
#SBATCH --output=logs/ate_analysis/%A/%a/run.out   # %A = job ID, %a = array index
#SBATCH --error=logs/ate_analysis/%A/%a/run.err
#SBATCH -t 0-00:30:00      # Adjust the wall time as needed
#SBATCH --array=0-99         # This creates 100 array tasks (indices 0 to 99)
# Fetch data from the config.ini file

module purge

source <(grep = config.ini)

echo "=== CONFIG VALUES ===";
echo "DATA_DIR: $DATA_DIR";
echo "APPTAINER_IMG: $APPTAINER_IMG";
echo "ARGS: $@";
echo "======================";
echo;

apptainer exec $APPTAINER_IMG python simulations/run_simulation.py --random_state ${SLURM_ARRAY_TASK_ID} "$@"
