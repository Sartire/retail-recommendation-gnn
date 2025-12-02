# dataset class for subgraphs
import torch
from torch_geometric.data import Data

import numpy as np
import pandas as pd
import torch_geometric

from torch.utils.data import Dataset, DataLoader
#from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_undirected 
#from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

from datetime import datetime
from collections import deque, defaultdict
from collections.abc import Sequence

from pathlib import Path

# EdgeList with columns ['user_idx', 'item_idx', 'timestamp']
#type EdgeList = pd.DataFrame


class LinkSubgraphDataset(Dataset):
    def __init__(self, 
                 pos_edge_sample: pd.DataFrame,
                 neg_edge_sample: pd.DataFrame,
                 full_edge_data: pd.DataFrame,
                 graph_features: torch_geometric.data.Data,
                 hops = 1):

        self.pos_edges = pos_edge_sample
        self.neg_edges = neg_edge_sample
        self.full_edge_data = full_edge_data
        self.hops = hops
        self.graph_node_features = graph_features.x


        self.sample_edges = pd.concat([pos_edge_sample, neg_edge_sample], axis=0)
        self.labels = np.concatenate([
            np.ones(pos_edge_sample.shape[0], dtype=np.int64),
            np.zeros(neg_edge_sample.shape[0], dtype=np.int64)
        ], axis=0)

    def __len__(self):
        return self.sample_edges.shape[0]

    def __getitem__(self, idx):
        '''
        return a single sample to be passed into the network.
        '''
        # 
        y = self.labels[idx]

        subgraph_nodes, subgraph_edges, sub_edge_weights = self.extract_enclosing_subgraph(idx)

        return subgraph_nodes, subgraph_edges, sub_edge_weights, torch.tensor(y, dtype=torch.float)
    
    def drnl_labeling(self, sub_nodes, u_idx, v_idx, edge_index):
        '''
        Double-radius node labeling (DRNL) algorithm.
        Label based on distances to nodes in the enclosing subgraph.
        '''
        node_id_to_pos = {int(n): i for i, n in enumerate(sub_nodes.tolist())}
        num_sub_nodes = sub_nodes.size(0)
    
        ##Building adjacency list for subgraph
        adj = [[] for _ in range(num_sub_nodes)]
        ei = edge_index
        mask = torch.isin(ei[0], sub_nodes) & torch.isin(ei[1], sub_nodes)
        sub_e0 = ei[0][mask]
        sub_e1 = ei[1][mask]

        for a, b in zip(sub_e0.tolist(), sub_e1.tolist()):
            ia = node_id_to_pos[a]
            ib = node_id_to_pos[b]
            adj[ia].append(ib)
            adj[ib].append(ia)

        def bfs_dist(start_pos):
            dist = [float('inf')] * num_sub_nodes
            q = deque()
            dist[start_pos] = 0
            q.append(start_pos)
            while q:
                cur = q.popleft()
                for nb in adj[cur]:
                    if dist[nb] == float('inf'):
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
            return dist

        u_pos = node_id_to_pos[int(u_idx)]
        v_pos = node_id_to_pos[int(v_idx)]

        du_list = bfs_dist(u_pos)
        dv_list = bfs_dist(v_pos)

        du = torch.tensor(du_list, dtype=torch.float)
        dv = torch.tensor(dv_list, dtype=torch.float)

        big = 1e6
        du[torch.isinf(du)] = big
        dv[torch.isinf(dv)] = big

        du = du.long()
        dv = dv.long()

        dist_sum = du + dv
        dist_min = torch.min(du, dv)
        dist_sum_half = dist_sum // 2

        labels = 1 + dist_min + dist_sum_half * (dist_sum_half + 1)

        #u_pos_t = torch.tensor(u_pos, dtype=torch.long)
        #v_pos_t = torch.tensor(v_pos, dtype=torch.long)
        #labels[u_pos_t] = 1
        #labels[v_pos_t] = 1

        return labels
    
    def extract_enclosing_subgraph(self, idx):
        '''
        Extract the enclosing subgraph of a given edge.
        Only consider edges with earlier timestamps.
        '''
        curr_edge = self.sample_edges.iloc[idx]
        current_timestamp = curr_edge['timestamp']
        src = curr_edge['user_idx']
        dst = curr_edge['item_idx']
        
        # subset to edges with earlier timestamps
        # NOTE this prevents the edge to be predicited from appearing in the induced subgraph. 
        edge_history_df = self.full_edge_data[self.full_edge_data['timestamp'] < current_timestamp].reset_index(drop=True)

        #create tensor of the historical indicies
        # add self edges to prevent the anchor nodes from being out of bounds

        src_history = torch.tensor(np.concatenate([[src], [dst], edge_history_df['user_idx'].to_numpy()]), dtype=torch.long)
        dst_history = torch.tensor(np.concatenate([[src], [dst], edge_history_df['item_idx'].to_numpy()]), dtype=torch.long)
        edge_history, edge_weights = to_undirected(torch.stack([src_history, dst_history], dim=0),
                                     edge_attr=torch.tensor(edge_history_df['weight'].to_numpy(), dtype=torch.float))

        # extract enclosing subgraph
        
        center_nodes = torch.tensor([src, dst], dtype=torch.long)

        node_idx, sub_edge_index, node_mapping, edge_mask  = k_hop_subgraph(node_idx=center_nodes,
                                                                            num_hops=self.hops,
                                                                            edge_index=edge_history,
                                                                            relabel_nodes=True,
                                                                            directed=False)
        
        #print(edge_mask)
        
        sub_edge_weights = edge_weights[edge_mask]

        node_idx = node_idx.clone()
        sub_edge_index = sub_edge_index.clone()
        
        drnl_labels = self.drnl_labeling(node_idx, src, dst, sub_edge_index)

        sub_node_features = self.graph_node_features[node_idx].clone()
        labels_norm = drnl_labels.view(-1, 1).float()
        nodes_with_drnl = torch.cat([sub_node_features, labels_norm], dim=1)

        return nodes_with_drnl, sub_edge_index, sub_edge_weights



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
