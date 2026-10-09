#!/bin/bash
#SBATCH --job-name=param_scan
#SBATCH --output=logs_600m1/scan_600_%A_%a.out
#SBATCH --error=logs_600m1/scan_600_%A_%a.err
#SBATCH --array=0-999%100
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# ================================================================
# CREATE ALL REQUIRED DIRECTORIES
# ================================================================
mkdir -p logs_600m1
mkdir -p results_600m1

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
python michel_parameter_scan.py --mode run --index "$INDEX"
