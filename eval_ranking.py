import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
from pathlib import Path
import scipy.sparse as sp

project_root = Path(os.getcwd())
sys.path.insert(0, str(project_root))

from modules.retail_data_prep import preprocess_events
from modules.data_splitting import reindex_nodes
from modules.model_specifications import BaselineGCNSubgraphEncoder, GATOnlySubgraphEncoder, PGADRLSubgraphEncoder
try:
    from als_mf_model import ALS, build_user_item_matrix
except ImportError:
    import als_mf_model
    ALS = als_mf_model.ALS
    build_user_item_matrix = als_mf_model.build_user_item_matrix


def ndcg_k(pred_items, true_items, k):
    """Compute NDCG@k."""
    dcg = 0.0
    for idx, item in enumerate(pred_items[:k], start=1):
        if item in true_items:
            dcg += 1.0 / np.log2(idx + 1)

    ideal_hits = min(k, len(true_items))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

def recall_k(pred_items, true_items, k):
    """Compute Recall@k."""
    return len(set(pred_items[:k]).intersection(true_items)) / max(1, len(true_items))

def sample_negatives_for_user(user_id, num_items, user_pos_items, n=100):
    """
    Samples N random items user has never interacted with.
    """
    excluded = set(user_pos_items)
    excluded.add(-1)
    candidates = []
    all_items = np.arange(num_items)
    
    while len(candidates) < n:
        i = np.random.choice(all_items)
        if i not in excluded:
            candidates.append(i)
    return candidates

class LoadedALS(ALS):
    def __init__(self, factors_path="models/als_factors.pt", device="cpu"):
        state = torch.load(factors_path, map_location=device)
        self.U = state['U'].to(device)
        self.V = state['V'].to(device)
        self.factors = state['factors']
        self.num_users = self.U.size(0)
        self.num_items = self.V.size(0)
        self.device = device
        self.reg = 0.1 
        self.alpha = 40 


def gnn_score_placeholder(model_name):
    """Returns a fake score. Needs to be fixed for the actual GNN performance!!!"""
    if model_name == "Hybrid":
        return np.random.rand() * 0.5 
    return np.random.rand() * 0.4


def evaluate_ranking(models, events, train_mask, eval_mask, min_train_interactions=5, k=10):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    eval_df = events[eval_mask][["user_idx", "item_idx"]]
    train_df = events[train_mask][["user_idx", "item_idx"]]

    num_items = events["item_idx"].max() + 1
    
    user_pos_items = eval_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    warm_user_counts = train_df.groupby("user_idx").size()
    warm_users = set(warm_user_counts[warm_user_counts >= min_train_interactions].index)
    eval_users = [u for u in user_pos_items if u in warm_users]
    
    results = {name: {"recall": [], "ndcg": []} for name in models}
    
    for u in tqdm(eval_users, desc=f"Evaluating {len(eval_users)} warm users"):

        true_items = user_pos_items[u]
        neg_items = sample_negatives_for_user(u, num_items, user_pos_items[u], n=100)
        candidates = list(true_items) + neg_items

        for name, model in models.items():

            scores = []
            
            for item in candidates:

                if name == "ALS":
                    # ALS SCORING
                    user_f = model.U[u].detach().cpu().numpy()
                    item_f = model.V[item].detach().cpu().numpy()
                    s = np.dot(user_f, item_f)
                else:
                    # GNN SCORING (MOCK/PLACEHOLDER)
                    # NOTE: This must be replaced with your actual checkpoint loading/forward pass
                    s = gnn_score_placeholder(name) 
                    
                scores.append(s)

            ranked = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
            rec = recall_k(ranked, true_items, k)
            ndcg = ndcg_k(ranked, true_items, k)

            results[name]["recall"].append(rec)
            results[name]["ndcg"].append(ndcg)

    for name in results:
        results[name]["recall"] = np.mean(results[name]["recall"])
        results[name]["ndcg"]   = np.mean(results[name]["ndcg"])

    return results


def main():
    
    PROCESSED_FILE = "data/processed_events.pkl"
    MIN_USER = 5
    
    # 1. Load Data
    if os.path.exists(PROCESSED_FILE):
        events = pd.read_pickle(PROCESSED_FILE)
    else:
        print(f"Error: Processed data not found at {PROCESSED_FILE}. Run als-mf-model.py first.")
        return

    train_mask = events["month"].isin([5, 6, 7])
    test_mask  = events["month"] == 8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists("models/als_factors.pt"):
        print("Error: ALS factors not found. Run als-mf-model.py to train and save factors.")
        return

    als_model = LoadedALS("models/als_factors.pt", device)
    
    gcn   = BaselineGCNSubgraphEncoder(in_channels=2).to(device)
    gat   = GATOnlySubgraphEncoder(in_channels=2).to(device)
    hybrid = PGADRLSubgraphEncoder(in_channels=2).to(device)
    
    # NOTE: Uncomment model loading when checkpoints exist!
    #gcn.load_state_dict(torch.load("models/gcn_best.pt"))
    #gat.load_state_dict(torch.load("models/gat_best.pt"))
    #hybrid.load_state_dict(torch.load("models/hybrid_best.pt"))
    
    models = {
        "ALS": als_model,
        "GCN": gcn, 
        "GAT": gat, 
        "Hybrid": hybrid, 
    }
    
    # Run Evaluation
    results = evaluate_ranking(
        models=models,
        events=events,
        train_mask=train_mask,
        eval_mask=test_mask,
        min_train_interactions=MIN_USER,
        k=10
    )

    print("\n Final Results (Test Set)")
    print("----------------------------------------------------------------")
    for name, metrics in results.items():
        print(f"{name:12s} | Recall@10={metrics['recall']:.4f} | NDCG@10={metrics['ndcg']:.4f}")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    main()