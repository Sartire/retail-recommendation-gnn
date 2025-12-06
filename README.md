# Retail Recommendation Systems via Graph Neural Networks

For University of Virginia MSDS 6050: Deep Learning

* Alex DeLuca
* Veda Raghu
* Liam Donoghue


# How to use this repository (at UVA)

All provided commands are from this working directory.

## Setup:

Git clone this repository into a directory of your choice. 

The torch_geometric package needs to find `libbz.so.1.0` in your `LD_LIBRARY_PATH` in order to import.
I symlinked `/usr/lib64/libbz2.so` to `~/.local/lib/libbz.so.1.0`. You will need to do the same.

Change the directories to your repository location and desired cache and output directories in these files:

    - run_batching.sh
    - mini_run_batching.sh (smaller job for syntax checking)
    - run_model_training.py
    - testing_utils/setup_python.sh

## Install and check packages



run `source ./testing_utils/setup_python.sh` to load modules, update LD_LIBRARY_PATH, install required python packages, and check that they imported correctly. 

## Preprocess and cache the data:

From any Rivanna terminal, create a slurm job to cache the data by running `sbatch run_batching.sh`

This will create subdirectories with the caches for each specification given in `data_splits.json` Runtime approx 40 mins per spec. 

## Train the models

From OpenOnDemand, launch an interactive desktop session on a GPU machine.

Make sure you specified the paths you need in run_model_training.py

In the terminal run `python run_model_training.py`
 
This will save state dictionaries and loss/performance measures in subdirectories of the specificied output directory. 

## Exfiltrating data and sharing results

Rivanna does not let you persist shared read permissions on your scratch directories. You should use UVA's Globus to create collections for the model outputs for collaborators to get those files. 

For this project, I did, however, just keep chmod 770 -ing my scratch directory for group members to get the data. :(

As a result, the eval_ranking.py which computes the scores of interest and the loss_plots.ipynb to generate visualizations are idiosyncratic and not hooked in to the rest of the folder management. 




