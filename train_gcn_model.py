import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from modules.model_specifications import BaselineGCNSubgraphEncoder
from modules.subgraph_dataclass import PreprocessedDataset, create_preprocessed_dataloader
import pickle
from pathlib import Path

from time import time, ctime

cache_data_dir = 'scratch/mcg4aw/retail_data/hops_1'

cache_base_path = Path(cache_data_dir)




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

BATCH_SIZE = 10

train_loader = create_preprocessed_dataloader(cache_base_path / 'split_train', batch_size=BATCH_SIZE, collate_fn=collate_subgraphs)
test_loader = create_preprocessed_dataloader(cache_base_path / 'split_test', batch_size=BATCH_SIZE, collate_fn=collate_subgraphs)
val_loader = create_preprocessed_dataloader(cache_base_path / 'split_val', batch_size=BATCH_SIZE, collate_fn=collate_subgraphs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.BCEWithLogitsLoss()

def run_epoch(loader, model, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    all_logits = []
    all_labels = []
    total_loss = 0.0

    for x, edge_index, batch, y in tqdm(loader):
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(x, edge_index, batch)
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


in_channels = 2  ##for the dnrl labels
num_epochs = 10

gcn_model = BaselineGCNSubgraphEncoder(in_channels=in_channels).to(device)

gcn_opt = torch.optim.Adam(gcn_model.parameters(), lr=1e-3)

starttime = time()

train_results = []
test_results = []

print("GCN Training")
for epoch in range(1, num_epochs + 1):
    train_loss, train_auc, train_f1, train_acc = run_epoch(train_loader, gcn_model, gcn_opt)
    test_loss, test_auc, test_f1, test_acc = run_epoch(test_loader, gcn_model, optimizer=None)
    print(ctime())
    print(f"[GCN] Epoch {epoch:02d} | Train loss={train_loss:.4f}, AUC={train_auc:.4f}, F1={train_f1:.4f}, ACC={train_acc:.4f}")
    print(f"Test loss={test_loss:.4f}, AUC={test_auc:.4f}, F1={test_f1:.4f}, ACC={test_acc:.4f}")

    train_results.append((train_loss, train_auc, train_f1, train_acc))
    test_results.append((test_loss, test_auc, test_f1, test_acc))

print(f'Finished in {(time() - starttime)/60} minutes')

pickle.dump(train_results, open(Path('./train_results.pkl'), 'wb'))
pickle.dump(test_results, open(Path('./test_results.pkl'), 'wb'))