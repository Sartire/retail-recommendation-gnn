import os
import sys
from pathlib import Path
import torch_geometric.data.data as pyg_internal
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import scipy.sparse as sp  
import torch_geometric  
import torch_geometric.data
import torch_geometric.data.data as pyg_data_internal
torch.serialization.add_safe_globals([
    torch_geometric.data.Data,
    pyg_data_internal.DataEdgeAttr,
])


project_root = Path(os.getcwd())
sys.path.insert(0, str(project_root))

from modules.model_specifications import (
    BaselineGCNSubgraphEncoder,
    GATOnlySubgraphEncoder,
    PGADRLSubgraphEncoder,
)
from torch_geometric.utils import k_hop_subgraph
# Extract an enclosing k-hop subgraph for evaluation scoring

def extract_subgraph_for_eval(user_id, item_id, hops, events, full_graph, num_users, device):
    """
    Extracts a k-hop enclosing subgraph for (user, item) during evaluation.
    This mirrors the training-time subgraph structure BUT without DRNL labels.
    """

    src = int(user_id)
    dst = int(item_id + num_users)   

    center_nodes = torch.tensor([src, dst], dtype=torch.long, device=device)

    # Run k-hop extraction on the full graph
    node_idx, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx=center_nodes,
        num_hops=hops,
        edge_index=full_graph.edge_index.to(device),
        relabel_nodes=True,
        directed=False,
    )

    # Gather node features
    x_sub = full_graph.x[node_idx].to(device)

    return x_sub, sub_edge_index.to(device)



def recall_k(pred_items, true_items, k=10):
    """Recall@k for one user."""
    return len(set(pred_items[:k]).intersection(true_items)) / max(1, len(true_items))


def ndcg_k(pred_items, true_items, k=10):
    """NDCG@k for one user."""
    dcg = 0.0
    for idx, item in enumerate(pred_items[:k], start=1):
        if item in true_items:
            dcg += 1.0 / np.log2(idx + 1)

    ideal_hits = min(k, len(true_items))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def sample_negatives_for_user(num_items, pos_items, n=100):
    """Sample n random items not in the user's positive set."""
    excluded = set(pos_items)
    excluded.add(-1)
    candidates = []

    while len(candidates) < n:
        i = np.random.randint(0, num_items)
        if i not in excluded:
            candidates.append(i)

    return candidates


def build_or_load_graph(events):
    graph_path = Path("data/full_graph_data.pt")

    if graph_path.exists():
        print("Loading cached full_graph_data.pt ...")
        return torch.load(graph_path, weights_only=False)

    print("Rebuilding full graph from events...")
    num_users = events["user_idx"].max() + 1
    num_items = events["item_idx"].max() + 1

    ui = torch.tensor([events["user_idx"].values,
                       events["item_idx"].values + num_users], dtype=torch.long)

    edge_index = torch_geometric.utils.to_undirected(ui)
    x = torch.zeros(num_users + num_items, 2).float()

    full_graph = torch_geometric.data.Data(x=x, edge_index=edge_index)

    torch.save(full_graph, graph_path)
    return full_graph




def extract_subgraph_for_pair(full_graph, user_id, item_id, num_users, hops):
    """
    Build a small enclosing subgraph around (user_id, item_node_id).
    We ignore time and DRNL here for simplicity and compatibility with
    the saved checkpoints (which expect in_channels=2).
    """
    # Map item_idx to global node index in bipartite graph
    center_user = user_id
    center_item = num_users + item_id

    center_nodes = torch.tensor([center_user, center_item], dtype=torch.long)
    node_idx, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx=center_nodes,
        num_hops=hops,
        edge_index=full_graph.edge_index,
        relabel_nodes=True,
        directed=False,
    )

    x_sub = full_graph.x[node_idx]  # [n_sub, 2]
    return x_sub, sub_edge_index




def score_gnn_model(model, hops, user_id, candidates, events, full_graph, num_users, device):

    model.eval()
    scores = []

    for item in candidates:

        # Extract subgraph
        x_sub, sub_edge_index = extract_subgraph_for_eval(
            user_id=user_id,
            item_id=item,
            hops=hops,
            events=events,
            full_graph=full_graph,
            num_users=num_users,
            device=device,
        )

        batch = torch.zeros(x_sub.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            pred = model(x_sub, sub_edge_index, batch)
            score = float(pred.squeeze().cpu())

        scores.append(score)

    return scores





def evaluate_models(models, events, train_mask, eval_mask, k=10):
    """
    models: dict name -> { "model": nn.Module, "hops": int }
    """
    device = next(iter(models.values()))["model"].device

    full_graph = build_or_load_graph(events)
    num_users = events["user_idx"].max() + 1
    num_items = events["item_idx"].max() + 1

    train_df = events[train_mask][["user_idx", "item_idx"]]
    eval_df = events[eval_mask][["user_idx", "item_idx"]]

    # Warm users: at least 5 interactions in training
    warm_counts = train_df.groupby("user_idx").size()
    warm_users = set(warm_counts[warm_counts >= 5].index)

    user_truth = eval_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    eval_users = [u for u in user_truth if u in warm_users]

    print(f"Evaluating {len(eval_users)} warm users...")

    results = {name: {"recall": [], "ndcg": []} for name in models}

    for u in tqdm(eval_users):
        true_items = user_truth[u]
        neg_items = sample_negatives_for_user(num_items, true_items, n=100)
        candidates = list(true_items) + neg_items

        for name, info in models.items():
            m = info["model"]
            hops = info["hops"]

            scores = score_gnn_model(
                model=m,
                hops=hops,
                user_id=u,
                candidates=candidates,
                events=events,
                full_graph=full_graph,
                num_users=num_users,
                device=device,
            )

            ranking = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
            rec = recall_k(ranking, true_items, k)
            ndcg = ndcg_k(ranking, true_items, k)

            results[name]["recall"].append(rec)
            results[name]["ndcg"].append(ndcg)

    for name in results:
        results[name]["recall"] = float(np.mean(results[name]["recall"]))
        results[name]["ndcg"]   = float(np.mean(results[name]["ndcg"]))

    return results




def load_trained_models(model_root, device):
    model_root = Path(model_root)
    models = {}

    for folder in sorted(model_root.glob("*")):
        if not folder.is_dir():
            continue

        hops = 1 if "hop1" in folder.name else 2

        gcn_path = folder / "gcn_model.pt"
        gat_path = folder / "gat_model.pt"
        pga_path = folder / "pga_model.pt"

        if gcn_path.exists():
            m = BaselineGCNSubgraphEncoder(in_channels=2).to(device)
            state = torch.load(gcn_path, map_location=device)
            m.load_state_dict(state, strict=False)
            m.device = device
            models[f"GCN_{folder.name}"] = {"model": m, "hops": hops}

        if gat_path.exists():
            m = GATOnlySubgraphEncoder(in_channels=2).to(device)
            state = torch.load(gat_path, map_location=device)
            m.load_state_dict(state, strict=False)
            m.device = device
            models[f"GAT_{folder.name}"] = {"model": m, "hops": hops}

        if pga_path.exists():
            m = PGADRLSubgraphEncoder(in_channels=2).to(device)
            state = torch.load(pga_path, map_location=device)
            # IMPORTANT FIX:
            m.load_state_dict(state, strict=False)
            m.device = device
            models[f"PGA_{folder.name}"] = {"model": m, "hops": hops}

    return models



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    pkl_path = Path("data/processed_events.pkl")

    if pkl_path.exists():
        print("Loading events from", pkl_path)
        events = pd.read_pickle(pkl_path)

    else:
        raise FileNotFoundError(
            "Neither data/processed_events.pkl nor data/processed_events.parquet "
            "were found. Run your preprocessing script first."
        )

    train_mask = events["month"].isin([5, 6, 7])
    test_mask  = events["month"] == 8

    print("Loading trained models from /scratch/mcg4aw/retail_output/ ...")
    models = load_trained_models("/scratch/mcg4aw/retail_output/", device)
    print(f"Loaded {len(models)} GNN variants:", ", ".join(models.keys()))

    if not models:
        raise RuntimeError("No models were found in /scratch/mcg4aw/retail_output/.")

    results = evaluate_models(
        models=models,
        events=events,
        train_mask=train_mask,
        eval_mask=test_mask,
        k=10,
    )

    os.makedirs("results", exist_ok=True)
    out_path = Path("results/gnn_ranking_results.csv")
    pd.DataFrame(results).T.to_csv(out_path)
    print("\nSaved ranking results to:", out_path.resolve())
    print(pd.DataFrame(results).T)


if __name__ == "__main__":
    main()
