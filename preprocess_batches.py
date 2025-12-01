import sys
from time import ctime, time
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
from tqdm import tqdm
from multiprocessing import cpu_count
import pickle
from collections import defaultdict
import ray
import os
import json




project_root = os.path.dirname(os.path.abspath(__file__))
#modules_path = os.path.join(project_root, "modules")
sys.path.insert(0, project_root)




import warnings
warnings.filterwarnings("ignore")

if not __name__ == "__main__":
    print("loaded libraries for worker process?")




from modules.subgraph_dataclass import LinkSubgraphDataset
from modules.retail_data_prep import preprocess_events, create_graph_features
from modules.data_splitting import get_split_subset





# display bytes in logs so we see how much data is written
def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


@ray.remote
def preprocess_single_item(idx, dataset, cache_dir):
    """Helper function for parallel preprocessing"""
    
    #print(f"Type of dataset_ref: {type(dataset_ref)}")
    #print(f"Value of dataset_ref: {dataset_ref}")
    #assert isinstance(dataset_ref, ray.ObjectRef), "Lost the ObjectRef!"

    #dataset = ray.get(dataset_ref)
    
    # Get preprocessed item
    item = dataset[idx]
    
    # Save to disk
    cache_path = Path(cache_dir) / f"item_{idx:06d}.pt"
    with open(cache_path, 'wb') as f:
        torch.save(item, f)
    
    return idx




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


    ray.init(num_cpus=num_workers,
             runtime_env={
                         "working_dir": project_root
                         }
    )

    # set umask for the created data

    old_mask = os.umask(0o007)

    # specifications for number of hops and how to split the data
    with open('data_splits.json', 'r') as f:
        specs = json.load(f)
        f.close()
    


    print(f'Loading data: {ctime()}')


    events = preprocess_events(min_user_interactions = min_user_interactions, min_item_interactions = min_item_interactions, limit = limit)


    visited_paths = set()
    for setting in specs.keys():
        print(f'Preprocessing begins for spec {setting}: {ctime()}') 

        spec = specs[setting]
        hops = spec['hops']
        neg_ratio = spec['neg_ratio']
        splits = spec['splits']

        for split in splits.keys():

            cache_dir = base_data_dir / setting / f'split_{split}'
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
                                                            neg_ratio=neg_ratio
            )

            graph_feature = create_graph_features(sample_events)

            print(f'Creating {split} dataset: {ctime()}')
            dataset = LinkSubgraphDataset(
                pos_sample,
                neg_sample,
                sample_events,
                graph_feature,
                hops = hops    
            )

            print(f'Begin parallel caching for {split}: {ctime()}')
            start = time()


            dataset_ref = ray.put(dataset)

            print(f"Type of dataset: {type(dataset)}")
            print(f"Type of dataset_ref: {type(dataset_ref)}")
            print(f"Is dataset_ref an ObjectRef? {isinstance(dataset_ref, ray.ObjectRef)}")
            assert isinstance(dataset_ref, ray.ObjectRef), f"Expected ObjectRef, got {type(dataset_ref)}"

            # Try to verify it's in the object store
            try:
                test_get = ray.get(dataset_ref)
                print(f"Successfully retrieved from object store, type: {type(test_get)}")
            except Exception as e:
                print(f"ERROR retrieving from object store: {e}")


            # Prepare arguments for parallel processing
            idx_list = [i for i in range(len(dataset))]

            print(f"Preprocessing {len(dataset)} items using {num_workers} workers...")
            print(f'Start time: {ctime()}')

            all_results = []

            # Process in parallel with progress bar
            parallel_batch_size = 100
            for i in tqdm(range(0, len(dataset), parallel_batch_size), desc="Batches"):

                assert isinstance(dataset_ref, ray.ObjectRef), "Lost the ObjectRef!"
                batch_inds = idx_list[i:i+parallel_batch_size]
                futures = [preprocess_single_item.remote(idx, dataset_ref, cache_dir) for idx in batch_inds]
                batch_results = ray.get(futures)

            # Process/save results immediately
                all_results.extend(batch_results)

            # Clean up after each parallel batch
                del futures, batch_results

            # Clean up when completely done
            del dataset_ref
            del all_results

            print(f"Preprocessing complete! Data saved to {cache_dir}")
            print(f'End time: {ctime()}')


            end = time()
            print(f'Finished parallel caching for {split} in {end - start} seconds: {ctime()}')

            visited_paths.add(cache_dir)


    print(f'Finished preprocessing: {ctime()}')
    ray.shutdown()
    os.umask(old_mask)

    print(f'Checking disk usage:')

    for path in visited_paths:

        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        print(f"Path: {path.as_posix()} | size: {format_bytes(total_size)} | time: {ctime()}")

    print(f'Finished checking disk usage: {ctime()}')

