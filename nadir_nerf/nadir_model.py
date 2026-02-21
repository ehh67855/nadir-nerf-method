
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Type

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import orientation_loss, pred_normal_loss, scale_gradients_by_distance_squared
from nerfstudio.models.depth_nerfacto import DepthNerfactoModel, DepthNerfactoModelConfig


@dataclass
class NadirModelConfig(DepthNerfactoModelConfig):
    """Nadir model configuration with a learnable ground surface."""

    _target: Type = field(default_factory=lambda: NadirModel)
    height_grid_resolution: int = 256
    """Resolution of the learned XY ground-height grid."""
    height_residual_scale: float = 1.0
    """Ground-height range as a multiple of scene Z-span above the floor."""
    learn_ground_surface: bool = True
    """Enable learning of a spatially-varying ground surface."""
    ground_gate_softness: float = 0.5
    """Meters over which density is smoothly suppressed below the learned floor; 0 = hard gate."""
    ground_gate_steepness: float = 15.0
    """Steepness factor for the smooth gate; higher is closer to a binary mask."""
    strict_eval_ground_clamp: bool = True
    """Apply a hard below-ground clamp at eval/export even when training uses a soft gate."""
    ground_floor_margin: float = 0.0
    """Extra margin above scene-box min Z; density is zeroed below max(ground, scene_min_z + margin)."""
    ground_quantile: float = 0.1
    """Target quantile for depth-supervised ground fitting (smaller -> lower envelope)."""
    ground_loss_mult: float = 0.05
    """Weight for depth-driven quantile loss on the learned ground surface."""
    ground_smoothness_loss_mult: float = 1e-4
    """Weight for TV smoothness regularization on the ground surface."""
    ground_below_weight_loss_mult: float = 0.05
    """Weight for suppressing rendered mass below the learned ground."""
    ground_clamp_to_scene_box: bool = True
    """Zero density outside the scene-box XY to prevent spill-over at the edges."""
    hard_floor_z: float | None = None
    """Optional absolute floor z (world units). If set, density is zeroed below this regardless of the learned ground."""
    gate_proposals: bool = False
    """Whether to height-gate proposal networks (disable to match depth-nerfacto sampling)."""
    proposal_gate_softness: float | None = None
    """Optional softness override for proposal gating; defaults to ground_gate_softness."""
    proposal_gate_steepness: float | None = None
    """Optional steepness override for proposal gating; defaults to ground_gate_steepness."""
    proposal_gate_clamp_xy: bool = False
    """Clamp proposal gating in XY; kept false by default to reduce edge pinching."""
    proposal_gate_clamp_inside: bool = False
    """Apply inside-AABB mask during proposal gating; kept false to avoid double clamping."""
    export_ground_filter: bool = False
    """If False, export-time ground filtering uses only the global floor (no learned height mask)."""


class NadirModel(DepthNerfactoModel):
    """Depth-Nerfacto variant augmented with a learnable height field."""

    config: NadirModelConfig

    def populate_modules(self):
        """Initialize base modules and register learned ground-surface components."""
        super().populate_modules()

        # Absolute floor derived from config or scene box; density is forbidden below this.
        scene_floor = float(self.scene_box.aabb[0, 2].item())
        scene_ceiling = float(self.scene_box.aabb[1, 2].item())
        scene_z_span = max(scene_ceiling - scene_floor, 1e-3)
        self.hard_floor_z = (
            float(self.config.hard_floor_z)
            if self.config.hard_floor_z is not None
            else scene_floor + float(self.config.ground_floor_margin)
        )

        grid_resolution = max(2, int(self.config.height_grid_resolution))
        self.ground_height_grid = Parameter(torch.full((1, 1, grid_resolution, grid_resolution), -6.0))
        self.ground_height_span = float(max(self.config.height_residual_scale, 1e-4)) * scene_z_span

        # Gate proposal networks so sampling avoids regions far below the learned ground.
        if self.config.gate_proposals:
            prop_softness = (
                self.config.proposal_gate_softness
                if self.config.proposal_gate_softness is not None
                else self.config.ground_gate_softness
            )
            prop_steepness = (
                self.config.proposal_gate_steepness
                if self.config.proposal_gate_steepness is not None
                else self.config.ground_gate_steepness
            )
            self.density_fns = [
                self._make_height_gated_density_fn(
                    fn,
                    clamp_inside=self.config.proposal_gate_clamp_inside,
                    clamp_xy=self.config.proposal_gate_clamp_xy,
                    softness=prop_softness,
                    steepness=prop_steepness,
                )
                for fn in self.density_fns
            ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        if self.config.learn_ground_surface and all(
            parameter is not self.ground_height_grid for parameter in param_groups["fields"]
        ):
            param_groups["fields"].append(self.ground_height_grid)
        return param_groups

    def _get_base_floor(self, positions: torch.Tensor) -> torch.Tensor:
        aabb = self.scene_box.aabb.to(device=positions.device, dtype=positions.dtype)
        floor_from_scene = aabb[0, 2:3] + self.config.ground_floor_margin
        floor_from_hard = positions.new_tensor(self.hard_floor_z).reshape(1)
        return torch.maximum(floor_from_scene, floor_from_hard)

    def _predict_ground_height(self, positions: torch.Tensor) -> torch.Tensor:
        """Predict per-sample ground Z from XY by bilinear lookup in a learned grid."""
        base_floor = self._get_base_floor(positions)
        if not self.config.learn_ground_surface:
            return base_floor

        aabb = self.scene_box.aabb.to(device=positions.device, dtype=positions.dtype)
        xy_min = aabb[0, :2]
        xy_extent = (aabb[1, :2] - aabb[0, :2]).clamp_min(1e-6)
        xy = positions[..., :2]
        xy_normalized = ((xy - xy_min) / xy_extent).clamp(0.0, 1.0)
        sampling_grid = (xy_normalized * 2.0 - 1.0).reshape(1, -1, 1, 2)

        height_grid = self.ground_height_grid.to(device=positions.device, dtype=positions.dtype)
        residual_logits = F.grid_sample(
            height_grid,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        residual_logits = residual_logits.reshape(-1).reshape(*positions.shape[:-1], 1)
        residual_height = torch.sigmoid(residual_logits) * positions.new_tensor(self.ground_height_span)

        learned_floor = base_floor + residual_height
        ceiling = aabb[1, 2:3]
        return torch.minimum(learned_floor, ceiling)

    def _height_gate(
        self,
        positions: torch.Tensor,
        *,
        clamp_xy: bool = True,
        softness: float | None = None,
        steepness: float | None = None,
    ) -> torch.Tensor:
        """Gate that suppresses density below the learned ground surface."""
        aabb = self.scene_box.aabb.to(device=positions.device, dtype=positions.dtype)
        floor = self._predict_ground_height(positions)
        delta = positions[..., 2:3] - floor
        gate_softness = self.config.ground_gate_softness if softness is None else softness
        gate_steepness = self.config.ground_gate_steepness if steepness is None else steepness
        if gate_softness <= 0:
            gate_z = (delta >= 0).to(positions.dtype)
        else:
            softness_t = positions.new_tensor(gate_softness)
            steepness_t = positions.new_tensor(gate_steepness)
            gate_z = torch.sigmoid(steepness_t * delta / softness_t)
        if not self.training and self.config.strict_eval_ground_clamp:
            gate_z = gate_z * (delta >= 0).to(positions.dtype)

        if clamp_xy and self.config.ground_clamp_to_scene_box:
            inside_xy = (positions[..., :2] >= aabb[0, :2]) & (positions[..., :2] <= aabb[1, :2])
            gate_xy = inside_xy.all(dim=-1, keepdim=True).to(positions.dtype)
        else:
            gate_xy = 1.0

        return gate_z * gate_xy

    def _apply_height_gate(
        self,
        positions: torch.Tensor,
        density: torch.Tensor,
        *,
        clamp_inside: bool = False,
        clamp_xy: bool = False,
        softness: float | None = None,
        steepness: float | None = None,
    ) -> torch.Tensor:
        gate = self._height_gate(
            positions,
            clamp_xy=clamp_xy,
            softness=softness,
            steepness=steepness,
        )
        if clamp_inside:
            aabb = self.scene_box.aabb.to(device=positions.device, dtype=positions.dtype)
            inside = ((positions >= aabb[0]) & (positions <= aabb[1])).all(dim=-1, keepdim=True).to(positions.dtype)
            return density * gate * inside

        return density * gate

    def _make_height_gated_density_fn(
        self,
        density_fn: Callable,
        *,
        clamp_inside: bool = True,
        clamp_xy: bool = True,
        softness: float | None = None,
        steepness: float | None = None,
    ) -> Callable:
        """Gate proposal network density functions to avoid sampling far below ground."""

        def gated_fn(positions: torch.Tensor, *args, **kwargs):
            base = density_fn(positions, *args, **kwargs)
            gated = self._apply_height_gate(
                positions,
                base,
                clamp_inside=clamp_inside,
                clamp_xy=clamp_xy,
                softness=softness,
                steepness=steepness,
            )
            return F.relu(gated)

        return gated_fn

    def _ground_quantile_depth_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        if not self.config.learn_ground_surface:
            return self.ground_height_grid.new_zeros(())
        if "depth_image" not in batch or "ray_samples_list" not in outputs or not outputs["ray_samples_list"]:
            return self.ground_height_grid.new_zeros(())

        ray_samples = outputs["ray_samples_list"][-1]
        origins = ray_samples.frustums.origins[..., 0, :]
        directions = ray_samples.frustums.directions[..., 0, :]

        termination_depth = batch["depth_image"].to(self.device)
        if not self.config.is_euclidean_depth and "directions_norm" in outputs:
            termination_depth = termination_depth * outputs["directions_norm"].to(self.device)

        valid = torch.isfinite(termination_depth) & (termination_depth > 0)
        if not torch.any(valid):
            return termination_depth.new_zeros(())

        termination_depth = torch.nan_to_num(termination_depth, nan=0.0, posinf=0.0, neginf=0.0)
        target_points = origins + directions * termination_depth
        target_ground = self._predict_ground_height(target_points)
        target_z = target_points[..., 2:3]

        quantile = min(max(float(self.config.ground_quantile), 1e-3), 1.0 - 1e-3)
        quantile_t = target_z.new_tensor(quantile)
        residual = target_z - target_ground
        pinball = torch.maximum(quantile_t * residual, (quantile_t - 1.0) * residual)

        return pinball.reshape(-1)[valid.reshape(-1)].mean()

    def _ground_smoothness_loss(self) -> torch.Tensor:
        if not self.config.learn_ground_surface:
            return self.ground_height_grid.new_zeros(())
        heights = torch.sigmoid(self.ground_height_grid)
        dx = heights[:, :, 1:, :] - heights[:, :, :-1, :]
        dy = heights[:, :, :, 1:] - heights[:, :, :, :-1]
        return dx.abs().mean() + dy.abs().mean()

    def _below_ground_weight_loss(self, outputs: Dict) -> torch.Tensor:
        if "weights_list" not in outputs or "ray_samples_list" not in outputs:
            return self.ground_height_grid.new_zeros(())

        weights = outputs["weights_list"][-1]
        ray_samples = outputs["ray_samples_list"][-1]
        positions = ray_samples.frustums.get_positions()
        ground = self._predict_ground_height(positions)
        below_ground_distance = torch.relu(ground - positions[..., 2:3])
        return (weights * below_ground_distance).sum(dim=-2).mean()

    def _eval_ground_visibility_mask(self, ray_bundle, depth: torch.Tensor) -> torch.Tensor:
        """Visibility mask used for eval/export to reject rendered points below floor."""
        rendered_points = ray_bundle.origins + ray_bundle.directions * depth
        if self.config.export_ground_filter:
            floor = self._predict_ground_height(rendered_points)
        else:
            floor = self._get_base_floor(rendered_points)
        return (rendered_points[..., 2:3] >= floor).to(depth.dtype)

    def get_outputs(self, ray_bundle):
        """Same as depth-nerfacto but gate densities with the learned ground surface."""
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        positions = ray_samples.frustums.get_positions()
        gated_density = self._apply_height_gate(
            positions,
            field_outputs[FieldHeadNames.DENSITY],
            clamp_inside=False,
            clamp_xy=False,
        )
        field_outputs[FieldHeadNames.DENSITY] = gated_density

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        if not self.training:
            accumulation = accumulation * self._eval_ground_visibility_mask(ray_bundle, depth)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }
        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

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

    def get_rgba_image(self, outputs: Dict[str, torch.Tensor], output_name: str = "rgb") -> torch.Tensor:
        """Return RGBA with true volumetric alpha for point-cloud export filtering."""
        accumulation_name = output_name.replace("rgb", "accumulation")
        if accumulation_name not in outputs:
            return super().get_rgba_image(outputs, output_name)
        rgb = outputs[output_name]
        acc = outputs[accumulation_name]
        if acc.dim() < rgb.dim():
            acc = acc.unsqueeze(-1)
        return torch.cat((rgb, acc.clamp(0.0, 1.0)), dim=-1)

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            metrics_dict["ground_quantile"] = self._ground_quantile_depth_loss(outputs, batch)
            metrics_dict["ground_smoothness"] = self._ground_smoothness_loss()
            metrics_dict["ground_below_weight"] = self._below_ground_weight_loss(outputs)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training and metrics_dict is not None:
            if self.config.ground_loss_mult > 0.0 and "ground_quantile" in metrics_dict:
                loss_dict["ground_quantile"] = self.config.ground_loss_mult * metrics_dict["ground_quantile"]
            if self.config.ground_smoothness_loss_mult > 0.0 and "ground_smoothness" in metrics_dict:
                loss_dict["ground_smoothness"] = (
                    self.config.ground_smoothness_loss_mult * metrics_dict["ground_smoothness"]
                )
            if self.config.ground_below_weight_loss_mult > 0.0 and "ground_below_weight" in metrics_dict:
                loss_dict["ground_below_weight"] = (
                    self.config.ground_below_weight_loss_mult * metrics_dict["ground_below_weight"]
                )
        return loss_dict

    def get_image_metrics_and_images(self, outputs, batch):
        return super().get_image_metrics_and_images(outputs, batch)
