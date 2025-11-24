#!/bin/bash


#SBATCH --time=1:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=10   # number of CPU cores per task
#SBATCH --partition=standard   # partition
#SBATCH -J "CreateRetailBatches"   # job name
#SBATCH --mail-user=mcg4aw@virginia.edu   # email address
#SBATCH --mail-type=ALL
#SBATCH --constraint=rivanna   # cluster
#SBATCH --account=shakeri_ds6050   # allocation name
#SBATCH --mem=6G

cd /scratch/mcg4aw/retail-recommendation-gnn

module purge
module load python/3.11.4
module load pytorch/2.7.0

pip install torch_geometric

python preprocess_batches.py --num_workers 10
