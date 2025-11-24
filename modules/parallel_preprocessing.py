import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

from subgraph_dataclass import LinkSubgraphDataset
from retail_data_prep import preprocess_events, create_graph_features
from data_splitting import get_split_subset

# mostly generated via Claude

class PreprocessedDataset(Dataset):
    """Fast dataset that loads preprocessed data from disk"""
    
    def __init__(self, cache_dir, weights_only = False):
        self.cache_dir = Path(cache_dir)
        self.weights_only = weights_only
        self.file_list = sorted(list(self.cache_dir.glob("*.pt")))
        
        if not self.file_list:
            raise ValueError(f"No preprocessed files found in {cache_dir}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        with open(self.file_list[idx], 'rb') as f:
            return torch.load(f, weights_only=self.weights_only)


def preprocess_single_item(args):
    """Helper function for parallel preprocessing"""
    idx, dataset, cache_dir = args
    
    # Get preprocessed item
    item = dataset[idx]
    
    # Save to disk
    cache_path = Path(cache_dir) / f"item_{idx:06d}.pt"
    with open(cache_path, 'wb') as f:
        torch.save(item, f)
    
    return idx


def preprocess_dataset_parallel(dataset, cache_dir, num_workers=None):
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
    
    # Prepare arguments for parallel processing
    args_list = [(i, dataset, cache_dir) for i in range(len(dataset))]
    
    print(f"Preprocessing {len(dataset)} items using {num_workers} workers...")
    
    # Process in parallel with progress bar
    with Pool(num_workers) as pool:
        list(
            #tqdm(
            pool.imap(preprocess_single_item, args_list),
            #total=len(dataset),
            #desc="Preprocessing"
        #)
        )
    
    print(f"Preprocessing complete! Data saved to {cache_dir}")


def create_preprocessed_dataloader(cache_dir, batch_size=32, **dataloader_kwargs):
    """
    Create a DataLoader from preprocessed data.
    
    Args:
        cache_dir: Directory containing preprocessed .pkl files
        batch_size: Batch size for DataLoader
        **dataloader_kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader instance
    """
    dataset = PreprocessedDataset(cache_dir)
    return DataLoader(dataset, batch_size=batch_size, **dataloader_kwargs)


# Example usage
if __name__ == "__main__":
    '''
    Testing the caching strategy
    '''


    
    print('Loading data...')
    events = preprocess_events()

    print('Getting a test split of 1000 events...')
    sample_events, pos_sample, neg_sample = get_split_subset(
        events,
        subset_col = "month",
        split_values = [9],
        pos_limit = 1000,
        neg_ratio=1
    )

    graph_feature = create_graph_features(sample_events)

    print('Creating dataset...')
    original_dataset = LinkSubgraphDataset(
        pos_sample,
        neg_sample,
        sample_events,
        graph_feature,
        hops = 2
    )
    
    # 2. Preprocess in parallel and save to disk (do this once)

    print("Preprocessing dataset in parallel...")
    
    cache_directory = "./temp/test_cache"
    
    num_workers = cpu_count()
    
    preprocess_dataset_parallel(
            original_dataset,
            cache_directory,
            num_workers=num_workers  # Adjust based on your CPU
        )
    

    def collate_subgraphs(batch):
        xs, eis, ys = zip(*batch)

        ys = torch.stack(ys, dim=0)

        new_x = []
        new_edge_index = []
        # track which subgraph the data is from
        new_batch = []

        node_offset = 0
        subgraph_offset = 0

        for x_sub, ei_sub, in zip(xs, eis):
            n = x_sub.size(0)
            new_x.append(x_sub)
            new_edge_index.append(ei_sub + node_offset)
            b_sub = torch.zeros(x_sub.size(0), dtype=torch.long)
            new_batch.append(b_sub + subgraph_offset)

            node_offset += n
            subgraph_offset = new_batch[-1].max().item() + 1

        new_x = torch.cat(new_x, dim=0)
        new_edge_index = torch.cat(new_edge_index, dim=1)
        new_batch = torch.cat(new_batch, dim=0)

        return new_x, new_edge_index, new_batch, ys
    # 3. Create a fast DataLoader from preprocessed data
    print("Test loading from cache:")
    dataloader = create_preprocessed_dataloader(
        cache_directory,
        batch_size=20,
        shuffle=True,
        #num_workers=2,
        collate_fn=collate_subgraphs
    )
    
    # 4. Use the DataLoader for training
    print("\nTesting DataLoader:")
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        pass
    print("\nDone!")
