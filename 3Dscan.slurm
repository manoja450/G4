#!/bin/bash
#SBATCH --job-name=scan
#SBATCH --output=logs_3Dscan/scan_%A_%a.out
#SBATCH --error=logs_3Dscan/scan_%A_%a.err
#SBATCH --array=0-999%100
# 41 x 34 = 1394 grid points. Each task runs BOTH modules (M1 + M2).
# SLURM usually caps a single array at ~1000 tasks, so submit in TWO parts:
#   sbatch 3Dscan.slurm
#   sbatch --array=0-393%50 --export=ALL,INDEX_OFFSET=1000 3Dscan.slurm
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# ================================================================
# CREATE ALL REQUIRED DIRECTORIES
# ================================================================
mkdir -p logs_3Dscan
mkdir -p results_3Dscan
mkdir -p data_m1

# ================================================================
# SET LD_LIBRARY_PATH
# ================================================================
export LD_LIBRARY_PATH=/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4H2O/lib:/usr/lib64/root:/home/manoja450/geant4-install/lib64:/home/manoja450/xerces-install/lib64

echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"

# ================================================================
# ACTIVATE CONDA ENVIRONMENT
# ================================================================
source /home/manoja450/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "Python: $(which python)"
python --version

# ================================================================
# GO TO MAC DIRECTORY AND RUN THE SCAN
# ================================================================
cd /home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4H2O/mac

INDEX=$(( SLURM_ARRAY_TASK_ID + ${INDEX_OFFSET:-0} ))
echo "Grid index: $INDEX (array task ${SLURM_ARRAY_TASK_ID} + offset ${INDEX_OFFSET:-0})"

python michel3Dscan.py --mode run --index "$INDEX"
