import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
from modules.model_specifications import BaselineGCNSubgraphEncoder, GATOnlySubgraphEncoder, PGADRLSubgraphEncoder
from modules.subgraph_dataclass import PreprocessedDataset, create_preprocessed_dataloader
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from time import time, ctime
import os


cache_data_dir = '/scratch/mcg4aw/retail_data'
output_dir = '/scratch/mcg4aw/retail_output'

old_umask =os.umask(0o007)

# SETUP -------------------------------------------------------
## configurations
BATCH_SIZE = 10
in_channels = 2  
num_epochs = 10
num_layers = 2
num_heads = 4
hidden_dim = 64
lr_start = 1e-3


cache_base_path = Path(cache_data_dir)

data_versions = [d.name for d in cache_base_path.iterdir()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.BCEWithLogitsLoss()

output_base_path = Path(output_dir)


def collate_subgraphs(batch):
    xs, eis, ews, ys = zip(*batch)

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
    new_edge_weights = torch.cat(ews, dim=0)
    new_batch = torch.cat(new_batch, dim=0)

    return new_x, new_edge_index, new_edge_weights, new_batch, ys






## Define Epoch function --------------------------
def run_epoch(loader, model, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    all_logits = []
    all_labels = []
    total_loss = 0.0

    for x, edge_index, edge_weights, batch, y in tqdm(loader):
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_weights = edge_weights.to(device)
        batch = batch.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(x, edge_index, edge_weights, batch)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs >= 0.5).astype(int)
    true = all_labels.numpy().astype(int)

    try:
        auc = roc_auc_score(true, probs)
    except ValueError:
        auc = np.nan

    f1 = f1_score(true, preds)
    acc = accuracy_score(true, preds)
    avg_loss = total_loss / len(true)

    return avg_loss, auc, f1, acc


def train_model_for_epochs(model, optimizer, scheduler, train_loader, test_loader, num_epochs, name='None'):
    train_results = []
    test_results = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_auc, train_f1, train_acc = run_epoch(train_loader, model, optimizer)
        test_loss, test_auc, test_f1, test_acc = run_epoch(test_loader, model, optimizer=None)
        print(ctime())
        print(f"[{name}] Epoch {epoch:02d} | Train loss={train_loss:.4f}, AUC={train_auc:.4f}, F1={train_f1:.4f}, ACC={train_acc:.4f}")
        print(f"Test loss={test_loss:.4f}, AUC={test_auc:.4f}, F1={test_f1:.4f}, ACC={test_acc:.4f}")

        train_results.append((train_loss, train_auc, train_f1, train_acc))
        test_results.append((test_loss, test_auc, test_f1, test_acc))
        scheduler.step()


    # combine the results into a dataframe
    train_loss, train_auc, train_f1, train_acc = zip(*train_results)
    test_loss, test_auc, test_f1, test_acc = zip(*test_results)

    results = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'train_loss': train_loss,
        'train_auc': train_auc,
        'train_f1': train_f1,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_acc': test_acc
    })

    return results


### Run training on each data version --------------------------
starttime = time()
results = dict()

for version in data_versions:
    print("--"*15)
    print(ctime())
    print(f"Training on {version}")
    print("--"*15)
    version_start = time()
    param_dir = output_base_path / version
    param_dir.mkdir(parents=True, exist_ok=True)
    # get the loaders:
    train_loader = create_preprocessed_dataloader(cache_base_path / version / 'split_train', batch_size=BATCH_SIZE, collate_fn=collate_subgraphs)
    test_loader = create_preprocessed_dataloader(cache_base_path / version / 'split_test', batch_size=BATCH_SIZE, collate_fn=collate_subgraphs)


    ## train models 1 at a time to reduce GPU usage
    ## GCN ----------------------
    
    gcn_model = BaselineGCNSubgraphEncoder(in_channels=in_channels, hidden_channels=hidden_dim, num_layers=num_layers).to(device)
    gcn_opt = torch.optim.AdamW(gcn_model.parameters(), lr=lr_start)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(gcn_opt, num_epochs)


    gcn_performance = train_model_for_epochs(gcn_model, gcn_opt, sched, train_loader, test_loader, num_epochs, name='GCN')
    gcn_performance['model'] = 'GCN'
    
    
    
    # clean up to reduce GPU memory usage

    params = gcn_model.state_dict()
    torch.save(params, param_dir / 'gcn_model.pt')
    del gcn_model
    del gcn_opt
    del params
    del sched
    torch.cuda.empty_cache()

    ## GAT -------------------------------

    gat_model = GATOnlySubgraphEncoder(in_channels=in_channels,
                                       hidden_channels=hidden_dim,
                                       num_layers=num_layers, 
                                       heads=num_heads).to(device) 
    gat_opt = torch.optim.AdamW(gat_model.parameters(), lr=lr_start)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(gat_opt, num_epochs)

    gat_performance = train_model_for_epochs(gat_model, gat_opt, sched, train_loader, test_loader, num_epochs, name='GAT')
    gat_performance['model'] = 'GAT'

    # clean up to reduce GPU memory usage
    params = gat_model.state_dict()
    torch.save(params, param_dir /'gat_model.pt')
    del gat_model
    del gat_opt
    del params
    torch.cuda.empty_cache() 
    
    ### PGA -----------------------

    pga_model = PGADRLSubgraphEncoder(in_channels=in_channels,
                                      hidden_channels=hidden_dim,
                                      num_layers=num_layers).to(device)
    pga_opt = torch.optim.AdamW(pga_model.parameters(), lr=lr_start)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(pga_opt, num_epochs)

    pga_performance = train_model_for_epochs(pga_model, pga_opt, sched, train_loader, test_loader, num_epochs, name='PGA')
    pga_performance['model'] = 'PGA'

    # clean up to reduce GPU memory usage
    params = pga_model.state_dict()
    torch.save(params, param_dir/'pga_model.pt')
    del pga_model
    del pga_opt
    del params
    torch.cuda.empty_cache()


    results[version] = pd.concat([#gcn_performance,
                                  #gat_performance,
                                pga_performance], axis=0)
    version_end = time()
    print(f"Finished {version} in {(version_end - version_start)/60} minutes")

print(f'Finished in {(time() - starttime)/60} minutes')
pickle.dump(results, open(output_base_path / 'results.pkl', 'wb'))


