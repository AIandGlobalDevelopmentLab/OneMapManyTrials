#!/usr/bin/env bash
#SBATCH -A naiss2024-5-450 -p alvis
#SBATCH -N 1
#SBATCH -C NOGPU
#SBATCH -t 0-05:00:00
# Fetch data from the config.ini file

module purge

source <(grep = ../config.ini)

echo "=== CONFIG VALUES ===";
echo "DATA_DIR: $DATA_DIR";
echo "APPTAINER_IMG: $APPTAINER_IMG";
echo;

apptainer exec $APPTAINER_IMG python write_hdf5.py