#!/bin/bash

module purge
module load gcc/11.4.0
module load openmpi/4.1.4
module load apptainer/1.3.4
module load python/3.11.4
module load pytorch/2.7.0

export LD_LIBRARY_PATH=/home/mcg4aw/.local/lib:$LD_LIBRARY_PATH

echo "Module setup complete"

cd /scratch/mcg4aw/retail-recommendation-gnn

pip install -r requirements.txt 

echo "Requirements installed"


python ./testing_utils/check_imports.py

echo "Imports checked"

