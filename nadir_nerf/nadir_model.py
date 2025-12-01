
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Type

import torch
from torch import nn

from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components.losses import orientation_loss, pred_normal_loss, scale_gradients_by_distance_squared
from nerfstudio.models.depth_nerfacto import DepthNerfactoModel, DepthNerfactoModelConfig
from nerfstudio.models.nerfacto import NerfactoModel


@dataclass
class NadirModelConfig(DepthNerfactoModelConfig):
    """Nadir Model Configuration with height field hooks."""

    _target: Type = field(default_factory=lambda: NadirModel)
    height_grid_resolution: int = 256
    """Resolution for potential grid-based priors (placeholder)."""
    height_residual_scale: float = 1.0
    """Scale applied to residual height predictions."""
    far_plane: float = 50.0
    """Clamp ray marching to a nearer far plane to avoid sampling deep below ground."""
    ground_gate_softness: float = 0.0
    """If >0, meters over which density is smoothly suppressed below the learned ground; 0 = hard gate."""
    ground_gate_steepness: float = 50.0
    """Steepness factor for the smooth gate; higher is closer to a binary mask."""
    ground_gate_offset: float = 0.0
    """Vertical offset (meters) applied to the learned ground height before gating."""
    ground_floor_margin: float = 0.0
    """Extra margin above scene-box min Z; density is zeroed below max(ground, scene_min_z + margin)."""
    ground_clamp_to_scene_box: bool = True
    """Zero density outside the scene-box XY to prevent spill-over at the edges."""
    hard_floor_z: float | None = None
    """Optional absolute floor z (world units). If set, density is zeroed below this regardless of the learned ground."""


class NadirModel(DepthNerfactoModel):
    """Depth-Nerfacto variant augmented with a learnable height field."""

    config: NadirModelConfig

    def populate_modules(self):
        """Initialize base modules then register the height residual network and gating."""
        super().populate_modules()
        # Absolute floor derived from config or scene box; used to forbid density below this plane.
        scene_floor = float(self.scene_box.aabb[0, 2].item())
        self.hard_floor_z = (
            float(self.config.hard_floor_z)
            if self.config.hard_floor_z is not None
            else scene_floor + float(self.config.ground_floor_margin)
        )
        self.height_residual_mlp = MLP(
            in_dim=2,
            out_dim=1,
            num_layers=3,
            layer_width=64,
            activation=nn.ReLU(),
            out_activation=None,
            implementation="torch",
        )
        # Gate proposal networks so sampling avoids regions far below the learned ground.
        self.density_fns = [self._make_height_gated_density_fn(fn) for fn in self.density_fns]

    def world_to_xy(self, positions: torch.Tensor) -> torch.Tensor:
        """Project world coordinates to the ground-plane XY."""
        return positions[..., :2]

    def base_height_field(self, xy: torch.Tensor) -> torch.Tensor:
        """Placeholder base height prior h0(x, y)."""
        return torch.zeros_like(xy[..., :1])

    def query_height(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute h(x, y) = h0(x, y) + delta_h(x, y) from the residual network."""
        xy = self.world_to_xy(positions)
        h0 = self.base_height_field(xy)
        delta_h = self.height_residual_mlp(xy)
        return h0 + self.config.height_residual_scale * delta_h

    def query_height_from_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """Expose a convenient XY-only height query for downstream export filters."""
        zeros = torch.zeros_like(xy[..., :1])
        positions = torch.cat([xy, zeros], dim=-1)
        return self.query_height(positions)

    def _height_gate(self, positions: torch.Tensor) -> torch.Tensor:
        """Gate that suppresses density below the learned ground height and outside the scene box."""
        aabb = self.scene_box.aabb.to(device=positions.device, dtype=positions.dtype)
        ground = self.query_height(positions) + self.config.ground_gate_offset
        # Enforce a hard floor at the scene-box minimum plus margin.
        floor = torch.maximum(ground, aabb[0, 2:3] + self.config.ground_floor_margin)
        delta = positions[..., 2:3] - floor
        if self.config.ground_gate_softness <= 0:
            gate_z = (delta >= 0).to(positions.dtype)
        else:
            softness = torch.tensor(self.config.ground_gate_softness, device=positions.device, dtype=positions.dtype)
            gate_z = torch.sigmoid(self.config.ground_gate_steepness * delta / softness)

        if self.config.ground_clamp_to_scene_box:
            inside_xy = (positions[..., :2] >= aabb[0, :2]) & (positions[..., :2] <= aabb[1, :2])
            gate_xy = inside_xy.all(dim=-1, keepdim=True).to(positions.dtype)
        else:
            gate_xy = 1.0

        return gate_z * gate_xy

    def _apply_height_gate(self, positions: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        gate = self._height_gate(positions)
        aabb = self.scene_box.aabb.to(device=positions.device, dtype=positions.dtype)
        inside = ((positions >= aabb[0]) & (positions <= aabb[1])).all(dim=-1, keepdim=True).to(positions.dtype)

        gated = density * gate * inside
        gated = torch.where(inside > 0.5, gated, torch.zeros_like(gated))
        return gated

    def _make_height_gated_density_fn(self, density_fn: Callable) -> Callable:
        """Gate proposal network density functions to avoid sampling far below ground."""

        def gated_fn(positions: torch.Tensor, *args, **kwargs):
            base = density_fn(positions, *args, **kwargs)
            gated = self._apply_height_gate(positions, base)
            return torch.nn.functional.relu(gated)

        return gated_fn

    def get_outputs(self, ray_bundle):
        """Same as depth-nerfacto but gate densities using the learned ground height."""
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        if ray_bundle.fars is not None:
            ray_bundle.fars = torch.minimum(
                ray_bundle.fars,
                torch.full_like(ray_bundle.fars, self.config.far_plane)
            )
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        positions = ray_samples.frustums.get_positions()
        gated_density = self._apply_height_gate(positions, field_outputs[FieldHeadNames.DENSITY])
        field_outputs[FieldHeadNames.DENSITY] = gated_density

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
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

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_rgba_image(self, outputs, output_name: str = "rgb"):
        """Use accumulation as the alpha channel so point-cloud export masks low-opacity rays."""
        accumulation_name = output_name.replace("rgb", "accumulation")
        if accumulation_name not in outputs:
            raise NotImplementedError(f"get_rgba_image is not implemented for model {self.__class__.__name__}")
        rgb = outputs[output_name]
        acc = outputs[accumulation_name]
        if acc.dim() < rgb.dim():
            acc = acc.unsqueeze(-1)
        return torch.cat((rgb, acc), dim=-1)

    # Depth supervision is optional. Fall back to vanilla Nerfacto metrics/losses
    # when the batch does not carry depth annotations.
    def get_metrics_dict(self, outputs, batch):
        if self.training and "depth_image" not in batch:
            return NerfactoModel.get_metrics_dict(self, outputs, batch)
        return super().get_metrics_dict(outputs, batch)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        if "depth_image" not in batch:
            return NerfactoModel.get_loss_dict(self, outputs, batch, metrics_dict)
        return super().get_loss_dict(outputs, batch, metrics_dict)

    def get_image_metrics_and_images(self, outputs, batch):
        if "depth_image" not in batch:
            return NerfactoModel.get_image_metrics_and_images(self, outputs, batch)
        return super().get_image_metrics_and_images(outputs, batch)
