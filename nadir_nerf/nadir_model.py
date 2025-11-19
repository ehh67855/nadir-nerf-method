
from __future__ import annotations

import types
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
    sheet_sigma: float = 0.5
    """Standard deviation of the Gaussian sheet prior."""
    sheet_amplitude: float = 1.0
    """Amplitude of the Gaussian sheet density contribution."""
    height_residual_scale: float = 1.0
    """Scale applied to residual height predictions."""


class NadirModel(DepthNerfactoModel):
    """Depth-Nerfacto variant augmented with a learnable height field."""

    config: NadirModelConfig

    def populate_modules(self):
        """Initialize base modules, then register the height residual network and density wrapper."""
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
        self._wrap_field_density()

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

    def get_gaussian_sheet_density(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian sheet density centered at the learned height."""
        if positions.numel() == 0:
            return torch.zeros_like(positions[..., :1])
        height = self.query_height(positions)
        z = positions[..., 2:3]
        signed_distance = z - height
        sigma = max(float(self.config.sheet_sigma), 1e-6)
        amplitude = float(self.config.sheet_amplitude)
        s_normalized = signed_distance / sigma
        return amplitude * torch.exp(-0.5 * (s_normalized**2))

    def _wrap_field_density(self) -> None:
        """Wrap the field density call with a Gaussian sheet additive prior."""
        original_get_density = self.field.get_density

        def wrapped_get_density(field_self, ray_samples, _orig=original_get_density):
            base_density, base_features = _orig(ray_samples)
            positions = ray_samples.frustums.get_positions()
            sheet_density = self.get_gaussian_sheet_density(positions)
            total_density = base_density + sheet_density
            return total_density, base_features

        self.field.get_density = types.MethodType(wrapped_get_density, self.field)
