
import torch
from torch import nn
#from torch.utils.data import Dataset, DataLoader
#from torch_geometric.data import Data
#from torch_geometric.utils import k_hop_subgraph, to_undirected
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import torch.nn.functional as F
#from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
#import matplotlib.pyplot as plt
#import seaborn as sns
#from datetime import datetime

class BaselineGCNSubgraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_weight,  batch):
        h = x
        #print(h.shape)
        #print(edge_index.shape)
        for conv in self.convs:
            h = conv(h, edge_index, edge_weight)
            #h = conv(h, edge_index)
            #print(h.shape)
            h = torch.relu(h)
        hg = global_mean_pool(h, batch)
        logit = self.mlp(hg).view(-1)
        return logit


class GATOnlySubgraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=False))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=heads, concat=False))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_weights, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_weights)
            #h = conv(h, edge_index)
            h = torch.relu(h)
        hg = global_mean_pool(h, batch)
        logit = self.mlp(hg).view(-1)
        return logit


class PGADRLSubgraphEncoder(nn.Module):
    """
    Hybrid GCN + GAT (PGA-DRL-style).
    Based on https://github.com/RS-Research/PGA-DRL/blob/main/Code/pga_drl.py
    """
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, heads=4):
        super().__init__()
        self.gcn_convs = nn.ModuleList()
        self.gcn_convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
        self.gat_convs = nn.ModuleList()
        self.gat_convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=False))
        for _ in range(num_layers - 1):
            self.gat_convs.append(GATConv(hidden_channels, hidden_channels, heads=heads, concat=False))

        embed_dim = 2 * hidden_channels
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        # Layer weights for progressive fusion
        self.layer_weights = nn.Parameter(torch.ones(num_layers + 1))

    def forward(self, x, edge_index, edge_weights, batch):
        gcn_x = x
        gcn_embeddings = []
        for conv in self.gcn_convs:
            gcn_x = conv(gcn_x, edge_index, edge_weights)
            #gcn_x = conv(gcn_x, edge_index)
            gcn_x = torch.relu(gcn_x)
            gcn_x =global_mean_pool(gcn_x, batch)
            gcn_embeddings.append(gcn_x)

        gat_x = x
        gat_embeddings = []
        for conv in self.gat_convs:
            gat_x = conv(gat_x, edge_index, edge_weights)
            #gat_x = conv(gat_x, edge_index)
            gat_x =  torch.relu(gat_x)
            gat_x = global_mean_pool(gat_x, batch)
            gat_embeddings.append(gat_x)


        gcn_stack = torch.stack(gcn_embeddings, dim=1)
        gat_stack = torch.stack(gat_embeddings, dim=1)

        layer_weights = F.softmax(self.layer_weights, dim=0)
        fused_embeddings = torch.sum((gcn_stack + gat_stack) * layer_weights.view(1, -1, 1), dim=1)

        # fuse the embeddings 
        #h = torch.cat([gcn_pool, gat_pool], dim=-1)
        logit = self.mlp(fused_embeddings).view(-1)
        return logit
