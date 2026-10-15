import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import os
import json
from collections import defaultdict
import random
from model_training.base_trainer import BaseTrainer


class BPRModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def forward(self, users, pos_items=None, neg_items=None):
        user_emb = self.user_embedding(users)
        if pos_items is not None and neg_items is not None:
            pos_emb = self.item_embedding(pos_items)
            neg_emb = self.item_embedding(neg_items)
            pos_scores = (user_emb * pos_emb).sum(dim=1)
            neg_scores = (user_emb * neg_emb).sum(dim=1)
            return pos_scores, neg_scores
        return user_emb


class BPRTrainer(BaseTrainer):
    """
    Bayesian Personalised Ranking trainer.
    Optimiser: AdamW (lr=0.02, weight_decay=0.01 as L2 regularisation).
    Loss: BPR-OPT = -log(sigmoid(score_pos - score_neg)).
    """

    def __init__(self, embedding_dim: int = 64, epochs: int = 50,
                 lr: float = 0.02, batch_size: int = 2048,
                 seed: int = 42, device: str = 'cuda'):
        super().__init__(embedding_dim, epochs, lr, batch_size, seed, device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.optimizer = None

    def _build_model(self, n_users: int, n_items: int) -> nn.Module:
        return BPRModel(n_users, n_items, self.embedding_dim)

    def _train_epoch(self, train_users: np.ndarray,
                     train_items: np.ndarray,
                     user_pos_items: dict,
                     epoch: int) -> float:
        """Train one epoch. Returns mean batch loss."""
        self.model.train()

        if self.optimizer is None:
            # AdamW with default weight_decay=0.01 provides L2 regularisation
            # on embeddings, equivalent to the λ_Θ||Θ||² term in BPR-OPT.
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=0.01
            )

        n_items     = len(self.item_map)
        dataset_size = len(train_users)
        indices      = torch.randperm(dataset_size)
        total_loss, n_batches = 0.0, 0

        pbar = tqdm(
            range(0, dataset_size, self.batch_size),
            desc=f'Epoch {epoch + 1:2d}/{self.epochs}',
            leave=False
        )

        for i in pbar:
            batch_idx   = indices[i: i + self.batch_size]
            batch_users = train_users[batch_idx.numpy()]
            batch_pos   = train_items[batch_idx.numpy()]

            # ── Vectorised negative sampling ─────────────────────────
            # Draw random negatives for the whole batch at once.
            # Only enter the per-user loop for the rare cases where the
            # drawn item is actually a positive for that user.
            # Collision probability ≈ |pos_u| / n_items.
            # MovieLens: ~80/3043 ≈ 2.6%  Amazon: ~20/38121 ≈ 0.05%
            batch_neg = np.random.randint(0, n_items, size=len(batch_users))

            for j, u in enumerate(batch_users):
                pos_set = user_pos_items.get(u, set())
                if batch_neg[j] in pos_set:
                    # Collision — resample until clean (rarely more than once)
                    neg = np.random.randint(0, n_items)
                    attempts = 0
                    while neg in pos_set and attempts < 50:
                        neg = np.random.randint(0, n_items)
                        attempts += 1
                    batch_neg[j] = neg

            users_t = torch.LongTensor(batch_users).to(self.device)
            pos_t   = torch.LongTensor(batch_pos).to(self.device)
            neg_t   = torch.LongTensor(batch_neg).to(self.device)

            self.optimizer.zero_grad()
            pos_scores, neg_scores = self.model(users_t, pos_t, neg_t)

            # BPR loss: -E[log σ(pos - neg)]
            loss = -torch.log(
                torch.sigmoid(pos_scores - neg_scores) + 1e-15
            ).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / n_batches

    def _batch_predict(self, user_indices: torch.Tensor,
                       item_indices: torch.Tensor) -> np.ndarray:
        """Efficient score matrix: [n_users, n_items]."""
        self.model.eval()
        with torch.no_grad():
            user_emb = self.model.user_embedding(user_indices)
            item_emb = self.model.item_embedding(item_indices)
            scores   = torch.mm(user_emb, item_emb.t())
        return scores.cpu().numpy()

    def get_user_embeddings(self) -> np.ndarray:
        return self.model.user_embedding.weight.detach().cpu().numpy()

    def get_item_embeddings(self) -> np.ndarray:
        return self.model.item_embedding.weight.detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train BPR model')
    parser.add_argument('--train',         type=str, required=True)
    parser.add_argument('--output',        type=str, default='./models/bpr')
    parser.add_argument('--epochs',        type=int, default=50)
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--batch-size',    type=int, default=2048)
    parser.add_argument('--lr',            type=float, default=0.02)
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--device',        type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    os.makedirs(args.output, exist_ok=True)

    trainer = BPRTrainer(
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device
    )
    trainer.fit(train_df)
    trainer.save_model(os.path.join(args.output, 'bpr_model.pth'))

    np.save(os.path.join(args.output, 'user_embeddings.npy'),
            trainer.get_user_embeddings())
    np.save(os.path.join(args.output, 'item_embeddings.npy'),
            trainer.get_item_embeddings())

    with open(os.path.join(args.output, 'mappings.json'), 'w') as f:
        json.dump({
            'user_map':     {str(k): v for k, v in trainer.user_map.items()},
            'item_map':     {str(k): v for k, v in trainer.item_map.items()},
            'item_map_rev': {str(k): v for k, v in trainer.item_map_rev.items()},
        }, f, indent=2)

    print(f"\n Model saved to {args.output}")