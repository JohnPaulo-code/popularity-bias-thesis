import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

from model_training.base_trainer import BaseTrainer


class NeuMFModel(nn.Module):
    """
    NeuMF model combining GMF and MLP components.

    Architecture
    ------------
    GMF path:
        user_emb_gmf [dim]  ⊙  item_emb_gmf [dim]  →  [dim]

    MLP path:
        concat(user_emb_mlp [dim], item_emb_mlp [dim])  →  [2*dim]
        → FC(2*dim → mlp_dims[0]) → ReLU
        → FC(mlp_dims[0] → mlp_dims[1]) → ReLU
        → ...
        → [mlp_dims[-1]]

    Output:
        concat(GMF_out, MLP_out)  →  FC  →  scalar score
    """

    def __init__(
            self,
            n_users:       int,
            n_items:       int,
            embedding_dim: int        = 64,
            mlp_dims:      list[int]  = None,
            dropout:       float      = 0.2,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [64, 32, 16, 8]

        self.embedding_dim = embedding_dim

        # ── GMF embeddings ────────────────────────────────────────────────
        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)

        # ── MLP embeddings ────────────────────────────────────────────────
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)

        # ── MLP layers ────────────────────────────────────────────────────
        mlp_layers  = []
        input_dim   = embedding_dim * 2          # concat of user + item
        for out_dim in mlp_dims:
            mlp_layers.append(nn.Linear(input_dim, out_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout))
            input_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # ── Output layer ──────────────────────────────────────────────────
        # Input = GMF output (embedding_dim) + MLP output (mlp_dims[-1])
        self.output_layer = nn.Linear(embedding_dim + mlp_dims[-1], 1, bias=False)

        # ── Initialisation ────────────────────────────────────────────────
        for emb in [self.user_embedding_gmf, self.item_embedding_gmf,
                    self.user_embedding_mlp, self.item_embedding_mlp]:
            nn.init.xavier_normal_(emb.weight)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_normal_(self.output_layer.weight)

    def _score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Compute scalar scores for (user, item) pairs — shape [batch]."""
        # GMF path
        u_gmf   = self.user_embedding_gmf(users)
        i_gmf   = self.item_embedding_gmf(items)
        gmf_out = u_gmf * i_gmf                          # element-wise product

        # MLP path
        u_mlp   = self.user_embedding_mlp(users)
        i_mlp   = self.item_embedding_mlp(items)
        mlp_out = self.mlp(torch.cat([u_mlp, i_mlp], dim=1))

        # Combine and score
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        score    = self.output_layer(combined).squeeze(1)
        return score

    def forward(
            self,
            users:     torch.Tensor,
            pos_items: torch.Tensor = None,
            neg_items: torch.Tensor = None,
    ):
        """
        Training: returns (pos_scores, neg_scores).
        Inference (no items given): returns user embeddings for matrix scoring.
        """
        if pos_items is not None and neg_items is not None:
            return self._score(users, pos_items), self._score(users, neg_items)

        # Inference mode — return None to trigger _batch_predict override
        return None

    # Expose standard embedding attributes so BaseTrainer._batch_predict
    # can fall back if needed (we override it anyway)
    @property
    def user_embedding(self):
        return self.user_embedding_gmf

    @property
    def item_embedding(self):
        return self.item_embedding_gmf


class NeuMFTrainer(BaseTrainer):
    """
    NeuMF trainer following the same interface as BPRTrainer and LightGCNTrainer.
    """

    def __init__(
            self,
            embedding_dim: int       = 64,
            mlp_dims:      list[int] = None,
            dropout:       float     = 0.2,
            epochs:        int       = 50,
            lr:            float     = 0.02,
            batch_size:    int       = 2048,
            seed:          int       = 42,
            device:        str       = "cuda",
    ):
        super().__init__(embedding_dim, epochs, lr, batch_size, seed, device)
        self.mlp_dims  = mlp_dims if mlp_dims is not None else [64, 32, 16, 8]
        self.dropout   = dropout
        self.optimizer = None

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _build_model(self, n_users: int, n_items: int) -> nn.Module:
        return NeuMFModel(
            n_users, n_items,
            embedding_dim=self.embedding_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout,
        )

    def _train_epoch(
            self,
            train_users:    np.ndarray,
            train_items:    np.ndarray,
            user_pos_items: dict,
            epoch:          int,
    ) -> float:
        """BPR training epoch."""
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
            batch_idx   = indices[i:i + self.batch_size].numpy()
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
        """
        Full score matrix for given users × all items.
        NeuMF scores every (user, item) pair through both GMF and MLP paths.
        Done in item-batches to avoid OOM on large item sets.
        """
        self.model.eval()
        n_users  = len(user_indices)
        n_items  = len(item_indices)
        scores   = torch.zeros(n_users, n_items, device=self.device)

        ITEM_CHUNK = 512   # process items in chunks to manage GPU memory

        with torch.no_grad():
            for start in range(0, n_items, ITEM_CHUNK):
                end        = min(start + ITEM_CHUNK, n_items)
                chunk_items = item_indices[start:end]          # [chunk]
                chunk_size  = len(chunk_items)

                # Expand to [n_users × chunk_size]
                u_exp = user_indices.unsqueeze(1).expand(n_users, chunk_size).reshape(-1)
                i_exp = chunk_items.unsqueeze(0).expand(n_users, chunk_size).reshape(-1)

                chunk_scores = self.model._score(u_exp, i_exp)  # [n_users * chunk_size]
                scores[:, start:end] = chunk_scores.view(n_users, chunk_size)

        return scores.cpu().numpy()