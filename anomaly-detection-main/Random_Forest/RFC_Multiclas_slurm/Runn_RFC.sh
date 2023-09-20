#!/bin/bash
#SBATCH --job-name=Runn_RFC
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH -A master # Replace with the desired account name
#SBATCH -p normal # Replace with the desired partition name
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --nodelist=hpc4 # Replace with the desired node name
# Additional SLURM options and job configuration
#SBATCH -t 1-05:00:00 # Set a time limit for the job (e.g., 10 minutes)
#SBATCH --mem-per-cpu=4G # Request memory per CPU (e.g.,4GB)
# Load any necessary modules or activate a virtual environment
source /cluster/datastore/kimkp/testven/bin/activate
# Commands to run your GPU job
srun python3 ./RF_gs_multiclass.py