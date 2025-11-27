import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm


from retail_data_prep import preprocess_events, create_graph_features
from data_splitting import get_split_subset
from functools import partial

import ray
import sys
import os

from multiprocessing import cpu_count
from time import ctime




@ray.remote
def preprocess_single_item(idx, dataset_ref, cache_dir):
    """Helper function for parallel preprocessing"""
     
    dataset = ray.get(dataset_ref)
    
    # Get preprocessed item
    item = dataset[idx]
    
    # Save to disk
    cache_path = Path(cache_dir) / f"item_{idx:06d}.pt"
    with open(cache_path, 'wb') as f:
        torch.save(item, f)
    
    return idx


def preprocess_dataset_parallel(dataset, cache_dir, num_workers=None, parallel_batch_size=10):
    """
    Preprocess entire dataset in parallel and save to disk.
    
    Args:
        dataset: Original Dataset instance with expensive __getitem__
        cache_dir: Directory to save preprocessed items
        num_workers: Number of parallel workers (default: cpu_count())
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if num_workers is None:
        num_workers = cpu_count()

    # ensure the workers can find the modules

    project_root = os.path.dirname(os.path.abspath(__file__))
    modules_path = os.path.join(project_root, "modules")
    sys.path.insert(0, modules_path)

# Import your class
    from subgraph_dataclass import LinkSubgraphDataset

    
    ray.init(num_cpus=num_workers,
             runtime_env={
                         "py_modules": [modules_path]
                         }
    )


    dataset_ref = ray.put(dataset)

    # Prepare arguments for parallel processing
    idx_list = [i for i in range(len(dataset))]
    
    print(f"Preprocessing {len(dataset)} items using {num_workers} workers...")
    print(f'Start time: {ctime()}')

    #futures = [preprocess_single_item.remote(idx, dataset, cache_dir) for idx in idx_list]
    
    all_results = []
    for i in tqdm(range(0, len(dataset), parallel_batch_size), desc="Batches"):
        

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

    ray.shutdown()



