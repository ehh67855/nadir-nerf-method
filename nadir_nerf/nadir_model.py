
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
    sheet_sigma: float = 0.30
    """Gaussian sheet thickness in meters (kept relatively wide)."""
    sheet_amplitude: float = 0.5
    """Small additive strength to avoid overpowering the base density."""
    height_residual_scale: float = 1.0
    """Scale applied to residual height predictions."""
    height_residual_clamp: float = 5.0
    """Clamp (meters) on residual predictions to prevent drift."""
    below_surface_loss_mult: float = 0.8
    """Weight for suppressing density below the surface."""
    above_surface_loss_mult: float = 0.2
    """Weight for suppressing tall columns above the canopy."""
    sheet_thinness_loss_mult: float = 0.05
    """Encourages sheet density to stay concentrated around the surface."""
    canopy_offset: float = 15.0
    """Maximum canopy height above the sheet in meters."""
    height_smoothness_mult: float = 0.001
    """Regularizes spatial smoothness of the height field."""


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
        self.height_residual_clamp = self.config.height_residual_clamp
        self._wrap_field_density()

    def world_to_xy(self, positions: torch.Tensor) -> torch.Tensor:
        """Project world coordinates to the ground-plane XY."""
        return positions[..., :2]

    def get_outputs(self, ray_bundle):
        """Attach height-pruning weights to stored ray samples while keeping Depth-Nerfacto outputs."""
        outputs = super().get_outputs(ray_bundle)
        if "ray_samples_list" in outputs:
            outputs["ray_samples_list"] = [
                self.prune_samples_by_height(ray_samples) for ray_samples in outputs["ray_samples_list"]
            ]
        return outputs

    def base_height_field(self, xy: torch.Tensor) -> torch.Tensor:
        """Placeholder base height prior h0(x, y)."""
        return torch.zeros_like(xy[..., :1])

    def query_height(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute h(x, y) = h0(x, y) + Δh(x, y) from the residual network."""
        xy = self.world_to_xy(positions)
        h0 = self.base_height_field(xy)
        delta_h = self.height_residual_mlp(xy)
        delta_h = torch.clamp(delta_h, -self.height_residual_clamp, self.height_residual_clamp)
        delta_h = self.config.height_residual_scale * delta_h
        return h0 + delta_h

    def get_gaussian_sheet_density(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian sheet density centered at the learned height."""
        if positions.numel() == 0:
            return torch.zeros_like(positions[..., :1])
        height = self.query_height(positions)
        z = positions[..., 2:3]
        s = z - height
        sigma = self.config.sheet_sigma
        amplitude = self.config.sheet_amplitude
        sheet_density = amplitude * torch.exp(-0.5 * (s / sigma) ** 2)
        return sheet_density

    def below_surface_loss(
        self, positions: torch.Tensor, total_density: torch.Tensor, weights: torch.Tensor, margin: float = 0.3
    ) -> torch.Tensor:
        """Penalize visible density that clearly falls below the learned surface."""
        h = self.query_height(positions)
        z = positions[..., 2:3]
        below_mask = z < (h - margin)
        visible_mask = weights > 1e-3
        mask = below_mask & visible_mask
        if not torch.any(mask):
            return torch.zeros((), device=positions.device, dtype=positions.dtype)
        penalty = total_density * weights
        return torch.mean(penalty[mask])

    def above_surface_loss(
        self, positions: torch.Tensor, total_density: torch.Tensor, weights: torch.Tensor, canopy_margin: float = 1.0
    ) -> torch.Tensor:
        """Penalize density well above a plausible canopy height and visible to the camera."""
        h = self.query_height(positions)
        z = positions[..., 2:3]
        canopy_limit = h + self.config.canopy_offset + canopy_margin
        above_mask = z > canopy_limit
        visible_mask = weights > 1e-3
        mask = above_mask & visible_mask
        if not torch.any(mask):
            return torch.zeros((), device=positions.device, dtype=positions.dtype)
        penalty = total_density * weights
        return torch.mean(penalty[mask])

    def sheet_thinness_loss(
        self, positions: torch.Tensor, sheet_density: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Discourage sheet density from spreading away from the surface where it actually renders."""
        h = self.query_height(positions)
        z = positions[..., 2:3]
        s = z - h
        active_mask = (sheet_density > 1e-3) & (weights > 1e-3)
        if not torch.any(active_mask):
            return torch.zeros((), device=positions.device, dtype=positions.dtype)
        loss = (s**2) * sheet_density * weights
        return torch.mean(loss[active_mask])

    def prune_samples_by_height(self, ray_samples):
        """Softly downweight samples far outside the acceptable height support."""
        positions = ray_samples.frustums.get_positions()
        h = self.query_height(positions)
        z = positions[..., 2:3]
        lower = h - 0.5
        upper = h + self.config.canopy_offset + 2.0
        below_penalty = torch.relu(lower - z)
        above_penalty = torch.relu(z - upper)
        penalty = below_penalty + above_penalty
        damp = torch.exp(-3.0 * penalty)
        if ray_samples.metadata is None:
            ray_samples.metadata = {}
        ray_samples.metadata["height_prune_weights"] = damp
        return ray_samples

    def height_smoothness_loss(self, xy: torch.Tensor) -> torch.Tensor:
        """Regularize the spatial gradients of the height field."""
        xy_var = xy.clone().detach().requires_grad_(True)
        zeros = torch.zeros_like(xy_var[..., :1])
        height_positions = torch.cat([xy_var, zeros], dim=-1)
        h = self.query_height(height_positions)
        grad = torch.autograd.grad(
            outputs=h,
            inputs=xy_var,
            grad_outputs=torch.ones_like(h),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return torch.mean(grad**2)

    def export_height_field_grid(self, resolution: int = 128) -> torch.Tensor:
        """Utility for visualizing the predicted height over the current scene bounds."""
        aabb = self.scene_box.aabb.to(self.device)
        min_xy = aabb[0, :2]
        max_xy = aabb[1, :2]
        xs = torch.linspace(min_xy[0], max_xy[0], resolution, device=self.device)
        ys = torch.linspace(min_xy[1], max_xy[1], resolution, device=self.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        grid = torch.stack([grid_x, grid_y], dim=-1)
        zeros = torch.zeros_like(grid[..., :1])
        positions = torch.cat([grid, zeros], dim=-1)
        return self.query_height(positions)

    def _wrap_field_density(self) -> None:
        """Wrap the field density call with a Gaussian sheet additive prior."""
        original_get_density = self.field.get_density

        def wrapped_get_density(field_self, ray_samples, _orig=original_get_density):
            ray_samples = self.prune_samples_by_height(ray_samples)
            base_density, base_features = _orig(ray_samples)
            positions = ray_samples.frustums.get_positions()
            sheet_density = self.get_gaussian_sheet_density(positions)
            total_density = base_density + sheet_density
            if ray_samples.metadata is not None:
                prune_weights = ray_samples.metadata.get("height_prune_weights")
            else:
                prune_weights = None
            if prune_weights is not None:
                total_density = total_density * prune_weights
            return total_density, base_features

        self.field.get_density = types.MethodType(wrapped_get_density, self.field)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Augment the base losses with geometry regularization terms."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if not self.training or "ray_samples_list" not in outputs or not outputs["ray_samples_list"]:
            return loss_dict

        ray_samples = outputs["ray_samples_list"][-1]
        positions = ray_samples.frustums.get_positions()
        total_density, _ = self.field.get_density(ray_samples)
        sheet_density = self.get_gaussian_sheet_density(positions)
        visibility = outputs["weights_list"][-1].detach()
        prune_weights = None
        if ray_samples.metadata is not None:
            prune_weights = ray_samples.metadata.get("height_prune_weights")

        loss_below = self.below_surface_loss(positions, total_density, visibility)
        loss_dict["below_surface"] = self.config.below_surface_loss_mult * loss_below

        loss_above = self.above_surface_loss(positions, total_density, visibility)
        loss_dict["above_surface"] = self.config.above_surface_loss_mult * loss_above

        loss_thin = self.sheet_thinness_loss(positions, sheet_density, visibility)
        loss_dict["sheet_thinness"] = self.config.sheet_thinness_loss_mult * loss_thin

        xy = positions[..., :2]
        loss_smooth = self.height_smoothness_loss(xy)
        loss_dict["height_smoothness"] = self.config.height_smoothness_mult * loss_smooth

        if prune_weights is not None:
            loss_dict["height_prune_reg"] = 0.001 * torch.mean((1.0 - prune_weights) * total_density)

        return loss_dict
