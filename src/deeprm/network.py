"""
CNN policy/value network for DeepRM_Plus.

Input  : state image of shape (C, H, W) where
         H = n_resources, W = time_horizon * (1 + n_visible).
Output : action logits of shape (action_dim,)  and  value scalar.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNPolicy(nn.Module):
    def __init__(self, in_channels: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1))
        # We'll use adaptive pooling so the head is independent of W.
        self.pool = nn.AdaptiveAvgPool2d((1, 16))
        self.fc1 = nn.Linear(32 * 16, hidden)
        self.fc_pi = nn.Linear(hidden, action_dim)
        self.fc_v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        h = F.relu(self.fc1(x))
        return self.fc_pi(h), self.fc_v(h).squeeze(-1)

    def policy(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]
