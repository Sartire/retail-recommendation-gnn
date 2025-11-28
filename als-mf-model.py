import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from tqdm import tqdm

class ALS:
    """
    PyTorch implementation of Implicit ALS (Hu, Koren, Volinsky 2008)
    """
    def __init__(self, num_users, num_items, factors=64, reg=0.01,
                 alpha=40, n_iters=10, device="cpu"):

        self.num_users = num_users
        self.num_items = num_items
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.n_iters = n_iters
        self.device = device

        self.U = torch.randn(num_users, factors, device=device) * 0.01
        self.V = torch.randn(num_items, factors, device=device) * 0.01

    def fit(self, user_item_csr):
        """
        Fit ALS factors using confidence-weighted implicit feedback.
        user_item_csr: scipy CSR matrix
        """
        print("Starting ALS training...")
        user_item_csr = user_item_csr.tocsr()

        I = torch.eye(self.factors, device=self.device)

        for it in range(self.n_iters):
            print(f"\nALS iteration {it+1}/{self.n_iters}")


            for u in tqdm(range(self.num_users)):
                row = user_item_csr[u]
                if row.nnz == 0:
                    continue

                item_idx = row.indices
                counts = torch.tensor(row.data, dtype=torch.float, device=self.device)

                Cu = 1 + self.alpha * counts
                Pu = (counts > 0).float()

                V_i = self.V[item_idx]
                CuI = torch.diag(Cu)

                A = V_i.T @ CuI @ V_i + self.reg * I
                b = V_i.T @ (CuI @ Pu)

                self.U[u] = torch.linalg.solve(A, b)

            for i in tqdm(range(self.num_items)):
                col = user_item_csr[:, i]
                if col.nnz == 0:
                    continue

                user_idx = col.indices
                counts = torch.tensor(col.data, dtype=torch.float, device=self.device)

                Cu = 1 + self.alpha * counts
                Pu = (counts > 0).float()

                U_u = self.U[user_idx]
                CuI = torch.diag(Cu)

                A = U_u.T @ CuI @ U_u + self.reg * I
                b = U_u.T @ (CuI @ Pu)

                self.V[i] = torch.linalg.solve(A, b)

    def recommend(self, user_id, N=10, exclude_items=None):
        """
        Generate top-N recommendations for a user.
        exclude_items : list of item indices to mask
        """
        scores = (self.U[user_id] @ self.V.T).detach().cpu().numpy()

        if exclude_items is not None:
            scores[exclude_items] = -1e9

        top_idx = np.argpartition(scores, -N)[-N:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return top_idx, scores[top_idx]




def build_user_item_matrix(events, num_users, num_items, weight_config):
    """
    Build CSR matrix of implicit feedback using event weights.
    """
    rows = events['user_idx'].values
    cols = events['item_idx'].values
    vals = events['event'].map(weight_config).astype(float).values

    return sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(num_users, num_items)
    )




def recall_ndcg_at_k(model, eval_df, train_df, K=10):
    """
    Compute user-level Recall@K and NDCG@K.
    """
    recall_scores = []
    ndcg_scores = []

    # group truth items per user
    true_items_dict = eval_df.groupby("user_idx")["item_idx"].apply(set)

    # which items users have already seen
    user_train_items = train_df.groupby("user_idx")["item_idx"].apply(set)

    for user, true_items in true_items_dict.items():

        seen = user_train_items.get(user, set())

        rec_items, _ = model.recommend(
            user_id=user,
            N=K,
            exclude_items=list(seen)
        )

        rec_items = list(rec_items)

        hits = len(true_items.intersection(rec_items))
        recall = hits / max(1, len(true_items))
        recall_scores.append(recall)

        dcg = 0
        for rank, item in enumerate(rec_items, start=1):
            if item in true_items:
                dcg += 1 / np.log2(rank + 1)

        ideal = min(K, len(true_items))
        idcg = sum(1 / np.log2(i + 1) for i in range(1, ideal + 1))

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(recall_scores), np.mean(ndcg_scores)



def main():
    events = pd.read_pickle("data/processed_events.pkl")



    num_users = events["user_idx"].max() + 1
    num_items = events["item_idx"].max() + 1

    train_df = events[events["month"].isin([5, 6])].copy()
    val_df   = events[events["month"] == 7].copy()
    test_df  = events[events["month"] == 8].copy()

    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))


    weight_cfg = {
        "view": 1.0,
        "addtocart": 2.0,
        "transaction": 3.0
    }

    print("Building user-item matrix...")
    train_matrix = build_user_item_matrix(
        train_df, num_users, num_items, weight_cfg
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training ALS on device: {device}")

    als = ALS(
        num_users=num_users,
        num_items=num_items,
        factors=64,
        reg=0.1,
        alpha=40,
        n_iters=5,
        device=device
    )

    als.fit(train_matrix)


    print("\nEvaluating ALS...")

    val_recall, val_ndcg = recall_ndcg_at_k(als, val_df, train_df, K=10)
    test_recall, test_ndcg = recall_ndcg_at_k(als, test_df, train_df, K=10)

    print("\n===== ALS RESULTS =====")
    print(f"Val Recall@10 = {val_recall:.4f}")
    print(f"Val NDCG@10   = {val_ndcg:.4f}")
    print(f"Test Recall@10 = {test_recall:.4f}")
    print(f"Test NDCG@10   = {test_ndcg:.4f}")

