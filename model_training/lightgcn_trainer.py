import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.sparse
from tqdm import tqdm
import random
from collections import defaultdict

from model_training.base_trainer import BaseTrainer


class LightGCNModel(nn.Module):
    """
    LightGCN with BPR training loss.

    The model stores only the learnable base embeddings (E^0).
    Graph propagation is performed on the fly during the forward pass
    using the pre-computed normalised adjacency matrix.
    """

    def __init__(
            self,
            n_users:       int,
            n_items:       int,
            embedding_dim: int = 64,
            n_layers:      int = 3,
    ):
        super().__init__()
        self.n_users       = n_users
        self.n_items       = n_items
        self.embedding_dim = embedding_dim
        self.n_layers      = n_layers

        # Learnable base embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        # Adjacency matrix set after construction via set_norm_adj()
        self.norm_adj: torch.sparse.FloatTensor = None

    def set_norm_adj(self, norm_adj: torch.sparse.FloatTensor):
        """Register the pre-computed normalised adjacency matrix."""
        self.norm_adj = norm_adj

    def _graph_propagation(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform n_layers of light graph convolution.
        Returns final user and item embeddings (mean-pooled across layers).
        """
        # Stack base embeddings: [n_users + n_items, dim]
        all_emb    = torch.cat([self.user_embedding.weight,
                                self.item_embedding.weight], dim=0)
        layer_embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            layer_embs.append(all_emb)

        # Mean pooling across layers (including layer 0)
        final_emb = torch.stack(layer_embs, dim=1).mean(dim=1)

        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        return user_emb, item_emb

    def forward(
            self,
            users:     torch.Tensor = None,
            pos_items: torch.Tensor = None,
            neg_items: torch.Tensor = None,
    ):
        """
        Training: returns (pos_scores, neg_scores) for BPR loss.
        Inference: returns (user_emb_all, item_emb_all) when called with no args.
        """
        user_emb_all, item_emb_all = self._graph_propagation()

        if users is not None and pos_items is not None and neg_items is not None:
            u_emb      = user_emb_all[users]
            p_emb      = item_emb_all[pos_items]
            n_emb      = item_emb_all[neg_items]
            pos_scores = (u_emb * p_emb).sum(dim=1)
            neg_scores = (u_emb * n_emb).sum(dim=1)
            return pos_scores, neg_scores

        return user_emb_all, item_emb_all


class LightGCNTrainer(BaseTrainer):
    """
    LightGCN trainer following the same interface as BPRTrainer.
    Trains with BPR loss on the interaction graph.
    """

    def __init__(
            self,
            embedding_dim: int = 64,
            n_layers:      int = 3,
            epochs:        int = 50,
            lr:            float = 0.02,
            batch_size:    int = 2048,
            seed:          int = 42,
            device:        str = "cuda",
    ):
        super().__init__(embedding_dim, epochs, lr, batch_size, seed, device)
        self.n_layers  = n_layers
        self.optimizer = None

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _build_model(self, n_users: int, n_items: int) -> nn.Module:
        return LightGCNModel(n_users, n_items, self.embedding_dim, self.n_layers)

    def fit(self, train_df, val_df=None):
        """
        Override fit() to build the normalised adjacency matrix after
        the model is constructed but before the training loop starts.
        BaseTrainer.fit() calls _train_epoch immediately after _build_model,
        which is too late — norm_adj would still be None on the first call.
        """
        # ── Replicate BaseTrainer.fit() setup ────────────────────────────
        self.user_map     = {uid: i for i, uid in enumerate(train_df["user_id"].unique())}
        self.item_map     = {iid: i for i, iid in enumerate(train_df["item_id"].unique())}
        self.item_map_rev = {v: k for k, v in self.item_map.items()}

        n_users = len(self.user_map)
        n_items = len(self.item_map)

        print(f"\n{'=' * 60}")
        print(f"Training {self.__class__.__name__}")
        print(f"{'=' * 60}")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Interactions: {len(train_df)}")
        print(f"Embedding dim: {self.embedding_dim}")

        self.train_users = train_df["user_id"].map(self.user_map).values
        self.train_items = train_df["item_id"].map(self.item_map).values

        self.user_pos_items = {}
        for u, i in zip(self.train_users, self.train_items):
            if u not in self.user_pos_items:
                self.user_pos_items[u] = set()
            self.user_pos_items[u].add(i)

        # Build model
        self.model = self._build_model(n_users, n_items).to(self.device)

        # ── Build and inject adj BEFORE training loop ─────────────────────
        norm_adj = self._build_norm_adj(
            self.train_users, self.train_items, n_users, n_items
        ).to(self.device)
        self.model.set_norm_adj(norm_adj)

        # ── Training loop ─────────────────────────────────────────────────
        for epoch in range(self.epochs):
            loss = self._train_epoch(
                self.train_users, self.train_items,
                self.user_pos_items, epoch
            )
            print(f"Epoch {epoch + 1:2d}/{self.epochs}: Loss = {loss:.4f}")

        self.training_complete = True
        print(f"\n Training complete!")
        return self

    def _build_norm_adj(
            self,
            train_users: np.ndarray,
            train_items: np.ndarray,
            n_users:     int,
            n_items:     int,
    ) -> torch.sparse.FloatTensor:
        """
        Build the symmetric normalised adjacency matrix used by LightGCN:

            A_hat = D^{-1/2} A D^{-1/2}

        where A is the (n_users + n_items) × (n_users + n_items) symmetric
        interaction matrix with user-item and item-user blocks.
        """
        n = n_users + n_items

        # Build COO indices for both directions
        # User → item block: row = user_idx, col = n_users + item_idx
        # Item → user block: row = n_users + item_idx, col = user_idx
        rows = np.concatenate([train_users,             n_users + train_items])
        cols = np.concatenate([n_users + train_items,   train_users])

        # Remove duplicates
        pairs = np.unique(np.stack([rows, cols], axis=1), axis=0)
        rows, cols = pairs[:, 0], pairs[:, 1]

        # Degree vector
        deg = np.bincount(rows, minlength=n).astype(np.float32)
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)

        # Normalised values: d_i^{-1/2} * d_j^{-1/2}
        values = deg_inv_sqrt[rows] * deg_inv_sqrt[cols]

        indices = torch.LongTensor(np.stack([rows, cols]))
        values  = torch.FloatTensor(values)

        norm_adj = torch.sparse_coo_tensor(indices, values, (n, n))
        return norm_adj.coalesce()

    def _train_epoch(
            self,
            train_users:    np.ndarray,
            train_items:    np.ndarray,
            user_pos_items: dict,
            epoch:          int,
    ) -> float:
        """BPR training epoch — identical structure to BPRTrainer."""
        self.model.train()

        if self.optimizer is None:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        dataset_size = len(train_users)
        indices      = torch.randperm(dataset_size)
        total_loss   = 0.0
        n_batches    = 0

        pbar = tqdm(range(0, dataset_size, self.batch_size),
                    desc=f"Epoch {epoch + 1}", leave=False)

        for i in pbar:
            batch_idx  = indices[i:i + self.batch_size].numpy()
            batch_users = train_users[batch_idx]
            batch_pos   = train_items[batch_idx]

            # Negative sampling
            batch_neg = np.random.randint(0, len(self.item_map), size=len(batch_users))
            for j, u in enumerate(batch_users):
                attempts = 0
                while batch_neg[j] in user_pos_items.get(u, set()) and attempts < 100:
                    batch_neg[j] = np.random.randint(0, len(self.item_map))
                    attempts += 1

            users_t = torch.LongTensor(batch_users).to(self.device)
            pos_t   = torch.LongTensor(batch_pos).to(self.device)
            neg_t   = torch.LongTensor(batch_neg).to(self.device)

            self.optimizer.zero_grad()
            pos_scores, neg_scores = self.model(users_t, pos_t, neg_t)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-15).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / n_batches

    def _batch_predict(
            self,
            user_indices: torch.Tensor,
            item_indices: torch.Tensor,
    ) -> np.ndarray:
        """Full-matrix prediction using propagated embeddings."""
        self.model.eval()
        with torch.no_grad():
            user_emb_all, item_emb_all = self.model()
            user_emb = user_emb_all[user_indices]
            item_emb = item_emb_all[item_indices]
            scores   = torch.mm(user_emb, item_emb.t())
        return scores.cpu().numpy()