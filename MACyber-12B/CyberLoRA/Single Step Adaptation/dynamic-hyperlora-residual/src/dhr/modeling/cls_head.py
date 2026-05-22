from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class LinearClsHead(nn.Module):
    """Single linear layer mapping mean-pooled hidden states to a binary logit."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: [B, hidden_dim] mean-pooled hidden states
        Returns:
            [B, 1] scalar logits (pre-sigmoid)
        """
        return self.fc(pooled)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hidden_dim": self.hidden_dim, "state_dict": self.state_dict()}, path)

    @staticmethod
    def load(path: str | Path, device: torch.device | str = "cpu") -> "LinearClsHead":
        payload: dict[str, Any] = torch.load(path, map_location=device)
        head = LinearClsHead(hidden_dim=int(payload["hidden_dim"]))
        head.load_state_dict(payload["state_dict"])
        return head.to(device)
