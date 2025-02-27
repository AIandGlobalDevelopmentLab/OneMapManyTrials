#!/usr/bin/env bash
#SBATCH -A naiss2024-5-450 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=V100:1  # We’re launching 1 node with 4 Nvidia A100 GPUs
#SBATCH -t 0-02:00:00
# Fetch data from the config.ini file

module purge

source <(grep = config.ini)

echo "=== CONFIG VALUES ===";
echo "DATA_DIR: $DATA_DIR";
echo "APPTAINER_IMG: $APPTAINER_IMG";
echo;

apptainer exec $APPTAINER_IMG python end_to_end_clip_training.py "$@"
