#!/usr/bin/env bash
#SBATCH -A naiss2024-5-450 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1  # We’re launching 1 node with 1 Nvidia T4 GPU
#SBATCH -t 0-02:00:00
# Fetch data from the config.ini file

module purge

source <(grep = ../config.ini)

echo "=== CONFIG VALUES ===";
echo "DATA_DIR: $DATA_DIR";
echo "APPTAINER_IMG: $APPTAINER_IMG";
echo;

apptainer exec $APPTAINER_IMG python train_model.py "$@"