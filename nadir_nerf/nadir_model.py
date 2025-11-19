
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import torch
from torch import nn

from nerfstudio.field_components.mlp import MLP
from nerfstudio.models.depth_nerfacto import DepthNerfactoModel, DepthNerfactoModelConfig


@dataclass
class NadirModelConfig(DepthNerfactoModelConfig):
    """Nadir Model Configuration with height field hooks."""

    _target: Type = field(default_factory=lambda: NadirModel)
    height_grid_resolution: int = 256
    """Resolution for potential grid-based priors (placeholder)."""
    sheet_sigma: float = 1.0
    """Std. deviation placeholder for future Gaussian sheet usage."""
    height_residual_scale: float = 1.0
    """Scale applied to residual height predictions."""


class NadirModel(DepthNerfactoModel):
    """Depth-Nerfacto variant augmented with a learnable height field."""

    config: NadirModelConfig

    def populate_modules(self):
        """Initialize base modules then register the height residual network."""
        super().populate_modules()
        self.height_residual_mlp = MLP(
            in_dim=2,
            out_dim=1,
            num_layers=3,
            layer_width=64,
            activation=nn.ReLU(),
            out_activation=None,
            implementation="torch",
        )

    def world_to_xy(self, positions: torch.Tensor) -> torch.Tensor:
        """Project world coordinates to the ground-plane XY."""
        return positions[..., :2]

    def base_height_field(self, xy: torch.Tensor) -> torch.Tensor:
        """Placeholder base height prior h0(x, y)."""
        return torch.zeros_like(xy[..., :1])

    def query_height(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute h(x, y) = h0(x, y) + Δh(x, y) from the residual network."""
        xy = self.world_to_xy(positions)
        h0 = self.base_height_field(xy)
        delta_h = self.height_residual_mlp(xy)
        return h0 + self.config.height_residual_scale * delta_h
