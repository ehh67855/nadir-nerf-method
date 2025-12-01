
from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from typing import Optional, Type

import torch
from torch import nn
import torch.nn.functional as F

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.models.depth_nerfacto import DepthNerfactoModel, DepthNerfactoModelConfig


@dataclass
class NadirModelConfig(DepthNerfactoModelConfig):
    """Nadir Model Configuration with height field hooks."""

    _target: Type = field(default_factory=lambda: NadirModel)
    height_grid_resolution: int = 256
    """Resolution for potential grid-based priors (placeholder)."""
    height_num_frequencies: int = 6
    """Number of Fourier feature frequencies for the XY height encoder."""
    surface_band_half_width: float = 0.25
    """Half-width (m) of the focused surface band to always sample."""
    num_surface_samples: int = 24
    """Samples tightly around the predicted surface height."""
    num_canopy_samples: int = 24
    """Samples between ground and canopy."""
    num_air_samples: int = 10
    """Sparse samples above canopy to enforce empty air."""
    num_diag_below_samples: int = 2
    """Tiny diagnostic samples below ground to enforce emptiness."""
    air_band_extra: float = 10.0
    """Extra meters above canopy to probe for air emptiness."""
    below_diag_margin: float = 0.2
    """How far below the surface to place diagnostic samples (m)."""
    sheet_sigma: float = 0.08
    """Gaussian sheet thickness in meters (tight membrane)."""
    sheet_amplitude: float = 1.5
    """Density boost on the surface membrane so the ground dominates."""
    height_residual_scale: float = 1.0
    """Scale applied to residual height predictions."""
    height_residual_clamp: float = 5.0
    """Clamp (meters) on residual predictions to prevent drift."""
    ground_gating_sharpness: float = 30.0
    """How sharply base density is suppressed below the learned surface."""
    ground_barrier_loss_mult: float = 0.2
    """Pushes signed distance to stay non-negative at visible points."""
    below_surface_loss_mult: float = 1e-4
    """Weight for suppressing density below the surface."""
    above_surface_loss_mult: float = 0.0
    """Weight for suppressing tall columns above the canopy (disabled)."""
    sheet_thinness_loss_mult: float = 0.0
    """Encourages sheet density to stay concentrated around the surface (disabled)."""
    canopy_offset: float = 15.0
    """Maximum canopy height above the sheet in meters."""
    height_smoothness_mult: float = 1e-6
    """Regularizes spatial smoothness of the height field (very small)."""
    height_anchor_mult: float = 1e-4
    """Anchor h(x,y) to prior h0(x,y) if available (very small)."""
    air_loss_mult: float = 0.0
    """Emptiness loss for far-air samples (disabled)."""
    canopy_sparsity_mult: float = 0.0
    """Penalize very tall densities well above canopy (disabled)."""
    air_monotonicity_mult: float = 0.0
    """Discourage density growth as we move far above the surface (disabled)."""
    sky_consistency_mult: float = 0.0
    """Penalize density along rays that mostly see sky (disabled)."""
    height_laplacian_mult: float = 0.0
    """Second-order smoothness for the height field to avoid ripples (disabled)."""
    air_height_margin: float = 6.0
    """How far above canopy to sample for air emptiness."""
    freeze_ground_steps: int = 1500
    """Number of warmup steps to keep ground frozen to the prior."""


class NadirModel(DepthNerfactoModel):
    """Depth-Nerfacto variant augmented with a learnable height field."""

    config: NadirModelConfig

    def populate_modules(self):
        """Initialize base modules, then register the height residual network and density wrapper."""
        # Capture optional dataset metadata (SfM/DEM prior) if provided via kwargs.
        self.metadata = self.kwargs.get("metadata", None)
        super().populate_modules()
        self.height_encoding_dim = 2 + 4 * self.config.height_num_frequencies
        self.height_residual_mlp = MLP(
            in_dim=self.height_encoding_dim,
            out_dim=1,
            num_layers=3,
            layer_width=64,
            activation=nn.ReLU(),
            out_activation=None,
            implementation="torch",
        )
        self.register_buffer("_height_prior_grid", None, persistent=False)
        self.register_buffer("_height_prior_valid", None, persistent=False)
        self._init_height_prior()
        self.height_residual_clamp = self.config.height_residual_clamp
        self._wrap_field_density()
        self._wrap_proposal_densities()

    def world_to_xy(self, positions: torch.Tensor) -> torch.Tensor:
        """Project world coordinates to the ground-plane XY."""
        return positions[..., :2]

    def _init_height_prior(self) -> None:
        """Optionally load an SfM/DEM prior height grid from metadata."""
        prior = None
        mask = None
        device = self.scene_box.aabb.device
        if self.metadata is not None:
            prior = self.metadata.get("height_prior_grid", None)
            mask = self.metadata.get("height_prior_valid", None)
        if prior is None:
            # Fallback: flat plane at scene_box bottom.
            z0 = self.scene_box.aabb[0, 2].detach()
            prior = torch.full(
                (self.config.height_grid_resolution, self.config.height_grid_resolution),
                z0,
                device=device,
            )
            mask = torch.ones_like(prior, dtype=torch.bool)
        self._height_prior_grid = prior.to(device)
        self._height_prior_valid = mask.to(device)

    def get_outputs(self, ray_bundle: RayBundle):
        """Custom forward pass that injects surface/canopy sampling and height-aware density gating."""
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)

        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )
        guided_ray_samples = self.height_guided_sampling(ray_bundle, ray_samples)
        ray_samples_list.append(guided_ray_samples)

        field_outputs = self.field.forward(guided_ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, guided_ray_samples)

        weights = guided_ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=guided_ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=guided_ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
            if self.config.predict_normals:
                outputs["rendered_orientation_loss"] = orientation_loss(
                    weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
                )
                outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                    weights.detach(),
                    field_outputs[FieldHeadNames.NORMALS].detach(),
                    field_outputs[FieldHeadNames.PRED_NORMALS],
                )

        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def base_height_field(self, xy: torch.Tensor) -> torch.Tensor:
        """Base height prior h0(x, y) via bilinear sampling on a prior grid."""
        if self._height_prior_grid is None:
            return torch.zeros_like(xy[..., :1])
        # Detach coords so we never backprop through grid_sample, avoiding unsupported gradients.
        xy_detached = xy.detach()
        aabb = self.scene_box.aabb.to(xy.device)
        min_xy = aabb[0, :2]
        max_xy = aabb[1, :2]
        # Normalize xy into [0, 1] for grid sampling.
        uv = (xy_detached - min_xy) / (max_xy - min_xy).clamp(min=1e-6)
        uv = uv.clamp(0.0, 1.0)
        grid = self._height_prior_grid
        # grid_sample expects shape (N, C, H, W)
        grid_4d = grid[None, None]
        # Build sampling coords in [-1, 1]
        coords = uv * 2 - 1
        coords = coords.view(1, -1, 1, 2)
        sampled = torch.nn.functional.grid_sample(
            grid_4d,
            coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        h0 = sampled.view(*xy.shape[:-1], 1)
        return h0

    def encode_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """Fourier-feature encode XY to let the height field capture sharper detail."""
        if self.config.height_num_frequencies <= 0:
            return xy
        freqs = 2.0 ** torch.arange(
            self.config.height_num_frequencies, device=xy.device, dtype=xy.dtype
        )
        scaled = xy[..., None, :] * freqs[:, None] * math.pi
        sin = torch.sin(scaled)
        cos = torch.cos(scaled)
        enc = torch.cat([sin, cos], dim=-1).reshape(*xy.shape[:-1], -1)
        return torch.cat([xy, enc], dim=-1)

    def query_height(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute h(x, y) = h0(x, y) + Δh(x, y) from the residual network."""
        xy = self.world_to_xy(positions)
        h0 = self.base_height_field(xy)
        encoded_xy = self.encode_xy(xy)
        delta_h = self.height_residual_mlp(encoded_xy)
        # Bound the residual with a tanh to act like a signed distance head.
        delta_h = torch.tanh(delta_h / max(self.height_residual_clamp, 1e-6)) * self.height_residual_clamp
        delta_h = self.config.height_residual_scale * delta_h
        raw_height = h0 + delta_h
        # Warmup: keep ground glued to h0 early in training.
        if self.training and getattr(self, "step", 0) < self.config.freeze_ground_steps:
            raw_height = h0.detach() + (raw_height - h0) * 0.0
        aabb = self.scene_box.aabb.to(positions.device)
        min_z, max_z = aabb[0, 2:3], aabb[1, 2:3]
        # Clamp to aabb with a small margin to avoid runaway planes.
        return torch.clamp(raw_height, min_z - 0.5, max_z + 0.5)

    def query_height_from_xy(self, xy: torch.Tensor, detach_prior: bool = False) -> torch.Tensor:
        """Query h(x,y) directly from 2D positions to avoid mixing z in the forward."""
        zeros = torch.zeros_like(xy[..., :1])
        pos = torch.cat([xy, zeros], dim=-1)
        h = self.query_height(pos)
        if detach_prior:
            h = h.detach()
        return h

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
        self, positions: torch.Tensor, raw_density: torch.Tensor, weights: torch.Tensor, margin: float = 0.0
    ) -> torch.Tensor:
        """Hard penalty for any density mass below the learned surface."""
        h = self.query_height(positions)
        z = positions[..., 2:3]
        signed = z - h
        below_mask = signed < -margin
        if not torch.any(below_mask):
            return torch.zeros((), device=positions.device, dtype=positions.dtype)
        base = raw_density
        penalty = torch.abs(base) + F.softplus(base * 5.0) * 0.2
        return torch.mean(penalty[below_mask])

    def ground_barrier_loss(
        self, positions: torch.Tensor, weights: torch.Tensor, margin: float = 0.05
    ) -> torch.Tensor:
        """Penalty that softly pushes signed distance to stay above the surface."""
        h = self.query_height(positions)
        s = positions[..., 2:3] - h
        visible_mask = weights > 1e-3
        if not torch.any(visible_mask):
            return torch.zeros((), device=positions.device, dtype=positions.dtype)
        barrier = torch.relu(-(s - margin))
        return torch.mean(barrier[visible_mask])

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
        loss = torch.abs(s) * sheet_density * weights
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
        damp = torch.exp(-8.0 * penalty)
        damp = damp * (z >= (h - self.config.below_diag_margin)).float()
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
        grad_mag = torch.sqrt(torch.sum(grad**2, dim=-1) + 1e-6)
        return torch.mean(grad_mag)

    def height_laplacian_loss(self, xy: torch.Tensor, eps: float = 0.5) -> torch.Tensor:
        """Second-order smoothness to keep the sheet locally planar."""
        if eps <= 0:
            return torch.zeros((), device=xy.device, dtype=xy.dtype)
        offsets = torch.tensor(
            [[eps, 0.0], [-eps, 0.0], [0.0, eps], [0.0, -eps]],
            device=xy.device,
            dtype=xy.dtype,
        )
        center_h = self.query_height_from_xy(xy)
        neigh_xy = xy[..., None, :] + offsets
        neigh_xy = neigh_xy.view(-1, 2)
        neigh_h = self.query_height_from_xy(neigh_xy)
        neigh_h = neigh_h.view(*xy.shape[:-1], 4, 1)
        lap = center_h[..., None, :] - neigh_h
        return torch.mean(torch.abs(lap))

    def _ray_ground_intersection(
        self, ray_bundle: RayBundle, use_prior_only: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-ray ground height and intersection distance along the ray."""
        origins = ray_bundle.origins
        directions = ray_bundle.directions
        xy0 = origins[..., :2]
        detach_prior = use_prior_only or (self.training and getattr(self, "step", 0) < self.config.freeze_ground_steps)
        h = self.query_height_from_xy(xy0, detach_prior=detach_prior)
        dir_z = directions[..., 2:3]
        safe_dir_z = torch.where(dir_z.abs() < 1e-4, dir_z.sign() + (dir_z == 0).float() * 1e-4, dir_z)
        t = (h - origins[..., 2:3]) / safe_dir_z
        xy_refined = xy0 + directions[..., :2] * t
        h = self.query_height_from_xy(xy_refined, detach_prior=detach_prior)
        t = (h - origins[..., 2:3]) / safe_dir_z
        return h, torch.clamp(t, min=0.0), safe_dir_z

    def _make_band_edges(
        self,
        height: torch.Tensor,
        origin_z: torch.Tensor,
        dir_z: torch.Tensor,
        offset_start: float,
        offset_end: float,
        num_samples: int,
    ) -> torch.Tensor:
        """Generate sorted t-edges for a band defined around the surface."""
        if num_samples <= 0:
            return torch.empty((*height.shape[:-1], 0), device=height.device, dtype=height.dtype)
        offsets = torch.linspace(offset_start, offset_end, num_samples + 1, device=height.device, dtype=height.dtype)
        offsets = offsets.view(*([1] * (height.ndim - 1)), -1, 1)
        z_edges = height[..., None, :] + offsets
        t_edges = (z_edges - origin_z[..., None, :]) / dir_z[..., None, :]
        t_edges = torch.sort(t_edges, dim=-2).values
        return torch.clamp(t_edges[..., 0], min=0.0)

    def height_guided_sampling(self, ray_bundle: RayBundle, base_ray_samples: RaySamples) -> RaySamples:
        """Merge proposal samples with focused surface/canopy/air sampling."""
        height, _, safe_dir_z = self._ray_ground_intersection(ray_bundle)
        origins = ray_bundle.origins

        base_edges = torch.cat(
            [base_ray_samples.frustums.starts[..., 0], base_ray_samples.frustums.ends[..., -1:, 0]], dim=-1
        )
        edges = [base_edges]

        surface_half = max(self.config.surface_band_half_width, self.config.sheet_sigma * 2.0)
        surface_edges = self._make_band_edges(
            height, origins[..., 2:3], safe_dir_z, -surface_half, surface_half, self.config.num_surface_samples
        )
        edges.append(surface_edges)

        canopy_edges = self._make_band_edges(
            height, origins[..., 2:3], safe_dir_z, 0.0, self.config.canopy_offset, self.config.num_canopy_samples
        )
        edges.append(canopy_edges)

        air_edges = self._make_band_edges(
            height,
            origins[..., 2:3],
            safe_dir_z,
            self.config.canopy_offset,
            self.config.canopy_offset + self.config.air_band_extra,
            self.config.num_air_samples,
        )
        edges.append(air_edges)

        if self.config.num_diag_below_samples > 0:
            below_edges = self._make_band_edges(
                height,
                origins[..., 2:3],
                safe_dir_z,
                -(self.config.surface_band_half_width + self.config.below_diag_margin),
                -self.config.below_diag_margin,
                self.config.num_diag_below_samples,
            )
            edges.append(below_edges)

        combined_edges = torch.cat(edges, dim=-1)
        combined_edges = torch.sort(combined_edges, dim=-1).values
        min_allowed = height - (self.config.surface_band_half_width + self.config.below_diag_margin)
        combined_edges = torch.maximum(combined_edges, min_allowed)
        combined_edges = torch.clamp(combined_edges, min=0.0)

        starts = combined_edges[..., :-1]
        ends = torch.maximum(combined_edges[..., 1:], starts + 1e-4)
        near = torch.min(starts, dim=-1, keepdim=True).values
        far = torch.max(ends, dim=-1, keepdim=True).values
        span = torch.clamp(far - near, min=1e-4)
        spacing_starts = (starts - near) / span
        spacing_ends = (ends - near) / span

        guided_samples = ray_bundle.get_ray_samples(
            bin_starts=starts[..., None],
            bin_ends=ends[..., None],
            spacing_starts=spacing_starts[..., None],
            spacing_ends=spacing_ends[..., None],
        )
        guided_samples.metadata = guided_samples.metadata or {}
        guided_samples.metadata["predicted_height"] = height.detach()
        return guided_samples

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
            # Minimal, differentiable soft gate applied to the final density.
            base_density, base_features = _orig(ray_samples)
            positions = ray_samples.frustums.get_positions()
            height = self.query_height(positions)
            signed = positions[..., 2:3] - height
            k = float(self.config.ground_gating_sharpness)
            # Soft sigmoid gate (no hard zeroing, no detach) to preserve gradients.
            if k > 0.0:
                gate = torch.sigmoid(k * signed)
            else:
                gate = torch.ones_like(signed)
            density = base_density * gate
            # Keep features untouched; do not zero features below ground.
            return density, base_features

        self._raw_density_fn = original_get_density
        self.field.get_density = types.MethodType(wrapped_get_density, self.field)

    def _wrap_proposal_densities(self) -> None:
        """Height-aware gating for proposal density functions so sampling ignores sub-ground space."""
        raw_fns = list(self.density_fns)
        masked_fns = []

        for fn in raw_fns:
            def _masked(positions, fn=fn):
                height = self.query_height_from_xy(
                    positions[..., :2],
                    detach_prior=self.training
                    and getattr(self, "step", 0) < self.config.freeze_ground_steps,
                )
                signed = positions[..., 2:3] - height
                gate = (signed >= 0.0).to(positions.dtype).detach()
                sharpness = self.config.ground_gating_sharpness
                if sharpness > 0:
                    if self.training:
                        warmup = min(
                            1.0, float(getattr(self, "step", 0)) / max(1.0, self.config.freeze_ground_steps)
                        )
                    else:
                        warmup = 1.0
                    gate = gate * torch.sigmoid(sharpness * warmup * torch.clamp(signed, min=0.0))
                return fn(positions) * gate

            masked_fns.append(_masked)

        self._raw_proposal_density_fns = raw_fns
        self.density_fns = masked_fns

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Augment the base losses with geometry regularization terms."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if not self.training or "ray_samples_list" not in outputs or not outputs["ray_samples_list"]:
            return loss_dict
        ray_samples = outputs["ray_samples_list"][-1]
        positions = ray_samples.frustums.get_positions()
        # raw_density comes from the unwrapped field get_density (before gate)
        raw_density, _ = self._raw_density_fn(ray_samples)

        # Lightweight below-ground regularizer (soft, uses raw density)
        signed = positions[..., 2:3] - self.query_height(positions)
        L_below = torch.mean(F.relu(-signed) * raw_density)
        loss_dict["below_surface"] = self.config.below_surface_loss_mult * L_below

        # Small smoothness and optional anchor to prior (very low weights)
        xy = positions[..., :2]
        loss_smooth = self.height_smoothness_loss(xy)
        loss_dict["height_smoothness"] = self.config.height_smoothness_mult * loss_smooth
        loss_anchor = self.height_anchor_loss(xy)
        loss_dict["height_anchor"] = self.config.height_anchor_mult * loss_anchor

        return loss_dict

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training and "ray_samples_list" in outputs and outputs["ray_samples_list"]:
            ray_samples = outputs["ray_samples_list"][-1]
            positions = ray_samples.frustums.get_positions()
            signed = positions[..., 2:3] - self.query_height(positions)
            metrics_dict["min_signed_distance"] = torch.min(signed).detach()
            # Diagnostic: mean of the soft ground gate
            k = float(self.config.ground_gating_sharpness)
            if k > 0.0:
                gate = torch.sigmoid(k * signed)
                metrics_dict["mean_ground_gate"] = torch.mean(gate).detach()
            metrics_dict["sheet_width_var"] = torch.var(signed).detach()
            canopy_violation = signed > (self.config.canopy_offset + 1.0)
            prune_w = torch.ones_like(signed)
            if ray_samples.metadata is not None and "height_prune_weights" in ray_samples.metadata:
                prune_w = ray_samples.metadata["height_prune_weights"]
            if torch.any(canopy_violation):
                metrics_dict["canopy_violation_density"] = torch.mean(prune_w[canopy_violation]).detach()
        return metrics_dict

    def height_anchor_loss(self, xy: torch.Tensor) -> torch.Tensor:
        """Anchor h(x,y) to the prior h0(x,y)."""
        if self._height_prior_grid is None:
            return torch.zeros((), device=xy.device, dtype=xy.dtype)
        h0 = self.base_height_field(xy).detach()
        valid = torch.ones_like(h0)
        if self._height_prior_valid is not None:
            grid = self._height_prior_valid.float()[None, None]
            aabb = self.scene_box.aabb.to(xy.device)
            xy_detached = xy.detach()
            uv = (xy_detached - aabb[0, :2]) / (aabb[1, :2] - aabb[0, :2]).clamp(min=1e-6)
            uv = uv.clamp(0.0, 1.0)
            coords = (uv * 2 - 1).view(1, -1, 1, 2)
            valid = torch.nn.functional.grid_sample(
                grid, coords, mode="bilinear", padding_mode="border", align_corners=True
            ).view_as(h0)
        height_positions = torch.cat([xy, torch.zeros_like(xy[..., :1])], dim=-1)
        h = self.query_height(height_positions)
        mask = (valid > 0.5).float()
        anchor = F.smooth_l1_loss(h, h0, reduction="none", beta=0.1)
        anchor = anchor * mask
        return torch.sum(anchor) / torch.clamp(mask.sum(), min=1.0)

    def air_loss(self, positions: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """Use existing far samples to enforce emptiness in high air regions."""
        h = self.query_height(positions)
        z = positions[..., 2:3]
        mask = z > (h + self.config.canopy_offset + self.config.air_height_margin)
        if not torch.any(mask):
            return torch.zeros((), device=positions.device, dtype=positions.dtype)
        return torch.mean(torch.abs(density[mask]))

    def canopy_sparsity_loss(self, positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Discourage very tall density columns above the canopy band."""
        h = self.query_height(positions)
        z = positions[..., 2:3]
        mask = z > (h + self.config.canopy_offset)
        if not torch.any(mask):
            return torch.zeros((), device=positions.device, dtype=positions.dtype)
        penalty = weights * (z - (h + self.config.canopy_offset))
        return torch.mean(penalty[mask])

    def air_monotonicity_loss(self, positions: torch.Tensor, densities: torch.Tensor) -> torch.Tensor:
        """Penalize density growth as we move far into the air."""
        h = self.query_height(positions)
        z = positions[..., 2:3]
        mask = z > (h + self.config.canopy_offset + 1.0)
        if densities.shape[-1] == 1:
            dens = densities[..., 0]
        else:
            dens = densities
        if dens.ndim < 2 or dens.shape[-1] < 2:
            return torch.zeros((), device=positions.device, dtype=positions.dtype)
        diffs = dens[..., 1:] - dens[..., :-1]
        mask_pair = (mask[..., 1:, 0] | mask[..., :-1, 0]).float()
        if not torch.any(mask_pair):
            return torch.zeros((), device=positions.device, dtype=positions.dtype)
        penalty = torch.relu(diffs) * mask_pair
        return torch.mean(penalty)

    def filter_point_cloud(self, points: torch.Tensor, density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-process a point cloud to drop any samples below the sheet and trim stray low-density clusters."""
        if points.numel() == 0:
            return points, density
        h = self.query_height_from_xy(points[..., :2], detach_prior=True)
        signed = points[..., 2:3] - h
        mask = (signed >= 0.0) & (density[..., :1] > 0)
        filtered_points = points[mask.squeeze(-1)]
        filtered_density = density[mask.squeeze(-1)]
        if filtered_density.numel() == 0:
            return filtered_points, filtered_density
        cutoff = torch.quantile(filtered_density, 0.1).detach()
        keep = filtered_density >= cutoff
        filtered_points = filtered_points[keep.squeeze(-1)]
        filtered_density = filtered_density[keep.squeeze(-1)]
        return filtered_points, filtered_density

    def sky_direction_loss(self, densities: torch.Tensor, accumulation: torch.Tensor, early_samples: int = 4):
        """If a ray mostly sees sky (low accumulation), force its early segments toward zero density."""
        if densities.numel() == 0:
            return torch.zeros((), device=accumulation.device, dtype=accumulation.dtype)
        sky_mask = accumulation < 0.05
        if densities.shape[-1] == 1:
            dens = densities[..., 0]
        else:
            dens = densities
        early = dens[..., : min(early_samples, dens.shape[-1])]
        mask = sky_mask
        while mask.ndim < early.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(early)
        if not torch.any(mask):
            return torch.zeros((), device=densities.device, dtype=densities.dtype)
        return torch.mean(torch.abs(early[mask]))
