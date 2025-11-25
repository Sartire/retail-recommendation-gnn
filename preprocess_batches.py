import sys
from time import ctime, time
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

if not __name__ == "__main__":
    print("loaded libraries for worker process?")


sys.path.append('./modules')

from subgraph_dataclass import LinkSubgraphDataset
from retail_data_prep import preprocess_events, create_graph_features
from data_splitting import get_split_subset
from parallel_preprocessing import preprocess_dataset_parallel

# display bytes in logs so we see how much data is written
def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


import argparse

parser = argparse.ArgumentParser(
    prog='preprocess_batches.py',
    description='''Preprocess retail rocket data in parallel.
    \nWrites the processed data to disk in cache directories.
    \nCaches are cleared if they already exist.
    \nCaches are in /scratch/mcg4aw/retail_data''',
)

parser.add_argument('-n','--num_workers', type=int, default=cpu_count(), help='Number of workers for parallel processing')
parser.add_argument('-d', '--base_cache_dir', type=str, default='./temp', help='Directory in which to save preprocessed items')
parser.add_argument('-l', '--limit', type=int, default=0, help='Limit the number of events? (0 for no limit)')
parser.add_argument('-mu', '--min_user_interactions', type=int, default=5, help='Minimum number of interactions for users')
parser.add_argument('-mi', '--min_item_interactions', type=int, default=10, help='Minimum number of interactions for items')

args = parser.parse_args()



if __name__ == "__main__":
    

    
    num_workers = args.num_workers
    base_data_dir = Path(args.base_cache_dir)
    limit = not args.limit == 0
    min_user_interactions = args.min_user_interactions
    min_item_interactions = args.min_item_interactions

    print(f'Number of workers: {num_workers}')
    print(f'Base cache directory: {base_data_dir}')
    print(f'Limit: {limit}')
    print(f'Minimum user interactions: {min_user_interactions}')
    print(f'Minimum item interactions: {min_item_interactions}')

    # specifications for number of hops and how to split the data
    specs = [{'hops': 2,
            'splits': {
                'train':[5,6,7],
                'test':[8],
                'val':[9]
                } 
            },
            ]


    print(f'Loading data: {ctime()}')


    events = preprocess_events(min_user_interactions = min_user_interactions, min_item_interactions = min_item_interactions, limit = limit)


    visited_paths = set()
    for i in range(len(specs)):  
        print(f'Preprocessing for spec {i + 1}: {ctime()}') 

        spec = specs[i]
        hops = spec['hops']
        splits = spec['splits']

        for split in splits.keys():

            cache_dir = base_data_dir / f'hops_{hops}' / f'split_{split}'
            # if the cache exists, clear it out
            if cache_dir.exists():
                print(f"Clearing out {cache_dir}: {ctime()}")
                for f in cache_dir.iterdir():
                    if f.is_file():
                        try:
                            f.unlink()  # Delete the file
                        except Exception as e:
                            print(f"Failed to delete {f}: {e}").unlink()

            cache_dir.mkdir(parents=True, exist_ok=True)

            sample_events, pos_sample, neg_sample = get_split_subset(
                                                            events,
                                                            subset_col = "month",
                                                            split_values = splits[split],
                                                            pos_limit = None,
                                                            neg_ratio=1
            )

            graph_feature = create_graph_features(sample_events)

            print(f'Creating {split} dataset: {ctime()}')
            original_dataset = LinkSubgraphDataset(
                pos_sample,
                neg_sample,
                sample_events,
                graph_feature,
                hops = hops
            )

            print(f'Begin parallel caching for {split}: {ctime()}')
            start = time()

            num_workers = min(num_workers, cpu_count()-2)
            preprocess_dataset_parallel(original_dataset, cache_dir, num_workers)

            end = time()
            print(f'Finished parallel caching for {split} in {end - start} seconds: {ctime()}')

            visited_paths.add(cache_dir)


    print(f'Finished preprocessing: {ctime()}')

    print(f'Checking disk usage:')

    for path in visited_paths:

        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        print(f"Path: {path.as_posix()} | size: {format_bytes(total_size)} | time: {ctime()}")

    print(f'Finished checking disk usage: {ctime()}')
