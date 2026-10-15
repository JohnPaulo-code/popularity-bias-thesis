from abc import ABC, abstractmethod
import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any


class BaseTrainer(ABC):
    def __init__(self, embedding_dim: int = 64, epochs: int = 20,
                 lr: float = 0.01, batch_size: int = 2048,
                 seed: int = 42, device: str = 'cuda'):
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

        # Set device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Model components (to be initialized in fit)
        self.model = None
        self.user_map: Dict[Any, int] = None
        self.item_map: Dict[Any, int] = None
        self.item_map_rev: Dict[int, Any] = None
        self.train_users = None
        self.train_items = None
        self.user_pos_items = None

        # For per-user metrics storage
        self.training_complete = False

    @abstractmethod
    def _build_model(self, n_users: int, n_items: int) -> torch.nn.Module:
        """Build the specific model architecture."""
        pass

    @abstractmethod
    def _train_epoch(self, train_users: np.ndarray, train_items: np.ndarray,
                     user_pos_items: Dict[int, set], epoch: int) -> float:
        """Train for one epoch. Return average loss."""
        pass


    def fit(self, train_df):
        """
        Train the model on training data.

        Args:
            train_df: DataFrame with columns ['user_id', 'item_id', 'rating', 'timestamp']
        """
        # Create mappings (0-based contiguous indices)
        self.user_map = {uid: i for i, uid in enumerate(train_df['user_id'].unique())}
        self.item_map = {iid: i for i, iid in enumerate(train_df['item_id'].unique())}
        self.item_map_rev = {v: k for k, v in self.item_map.items()}

        n_users = len(self.user_map)
        n_items = len(self.item_map)

        print(f"\n{'=' * 60}")
        print(f"Training {self.__class__.__name__}")
        print(f"{'=' * 60}")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Interactions: {len(train_df)}")
        print(f"Embedding dim: {self.embedding_dim}")

        # Convert data to indices
        self.train_users = train_df['user_id'].map(self.user_map).values
        self.train_items = train_df['item_id'].map(self.item_map).values

        # Build positive items lookup for negative sampling
        self.user_pos_items = {}
        for u, i in zip(self.train_users, self.train_items):
            if u not in self.user_pos_items:
                self.user_pos_items[u] = set()
            self.user_pos_items[u].add(i)

        # Build model
        self.model = self._build_model(n_users, n_items).to(self.device)

        # Training loop
        for epoch in range(self.epochs):
            loss = self._train_epoch(
                self.train_users, self.train_items,
                self.user_pos_items, epoch
            )
            print(f"Epoch {epoch + 1:2d}/{self.epochs}: Loss = {loss:.4f}")

        self.training_complete = True
        print(f"\n Training complete!")

        return self

    def predict_all_users(self, user_ids: List[Any]) -> Dict[str, Dict[str, float]]:
        """
        Generate full score predictions for specified users.

        Args:
            user_ids: List of original user IDs to predict for

        Returns:
            Dictionary: {user_id_str: {item_id_str: score, ...}}
        """
        if not self.training_complete:
            raise RuntimeError("Model must be trained before prediction")

        # Filter to users we have mappings for
        valid_users = [uid for uid in user_ids if uid in self.user_map]
        if len(valid_users) == 0:
            return {}

        # Get indices for valid users
        user_indices = torch.LongTensor([self.user_map[uid] for uid in valid_users]).to(self.device)
        all_items = torch.arange(len(self.item_map)).to(self.device)

        # Get scores (to be implemented by child classes if needed)
        scores = self._batch_predict(user_indices, all_items)

        # Convert to dictionary format
        predictions = {}
        for idx, uid in enumerate(valid_users):
            user_scores = {}
            for item_idx in range(len(self.item_map)):
                item_id = self.item_map_rev[item_idx]
                user_scores[str(item_id)] = float(scores[idx, item_idx])
            predictions[str(uid)] = user_scores

        return predictions

    def _batch_predict(self, user_indices: torch.Tensor,
                       item_indices: torch.Tensor) -> np.ndarray:
        """
        Batch prediction - can be overridden by child classes for optimization.
        Default implementation uses model's forward pass.

        Returns:
            numpy array of shape (len(user_indices), len(item_indices))
        """
        self.model.eval()
        with torch.no_grad():
            # Get embeddings
            user_emb = self.model.user_embedding(user_indices)
            item_emb = self.model.item_embedding(item_indices)

            # Compute all scores: [n_users, n_items]
            scores = torch.mm(user_emb, item_emb.t())

        return scores.cpu().numpy()

    def save_model(self, path: str):
        """Save model weights and mappings."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user_map': self.user_map,
            'item_map': self.item_map,
            'item_map_rev': self.item_map_rev,
            'embedding_dim': self.embedding_dim,
            'model_class': self.__class__.__name__
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model weights and mappings."""
        checkpoint = torch.load(path, map_location=self.device)

        self.user_map = checkpoint['user_map']
        self.item_map = checkpoint['item_map']
        self.item_map_rev = checkpoint['item_map_rev']
        self.embedding_dim = checkpoint['embedding_dim']

        n_users = len(self.user_map)
        n_items = len(self.item_map)

        self.model = self._build_model(n_users, n_items)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.training_complete = True

        print(f"Model loaded from {path}")
        return self