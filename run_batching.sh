#!/bin/bash

#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=20         
#SBATCH --mem-per-cpu=6000            
#SBATCH --time=01:00:00   
#SBATCH --chdir=/scratch/mcg4aw/retail-recommendation-gnn
#SBATCH --mail-user=mcg4aw@virginia.edu
#SBATCH --mail-type=ALL


python preprocess_batches.py $SLURM_CPUS_PER_TASK
