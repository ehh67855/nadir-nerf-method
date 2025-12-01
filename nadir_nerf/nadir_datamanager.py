from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)


@dataclass
class NadirDataManagerConfig(VanillaDataManagerConfig):
    #_ means used for internal use my the framework
    # This means, the value for target, should be a class type called NadirDataManager
    # dataclass annotation says that this functions defines how to instantiate instances of NaditDataManager
    _target: Type = field(default_factory=lambda: NadirDataManager)


class NadirDataManager(VanillaDataManager):

    config: NadirDataManagerConfig

    def __init__(
        self,
        config: NadirDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

    def to(self, device: Union[torch.device, str]):
        """Move datamanager state to `device`.

        The base `VanillaDataManager` in some nerfstudio versions provides a `.to` method
        but not all. Add a conservative implementation here so pipelines that call
        `datamanager.to(device)` won't fail. For child objects (datasets, samplers,
        ray generators) we try to call their `.to` if available.
        """
        self.device = torch.device(device)

        # Try moving common child objects if they implement `.to`
        for name in (
            "train_dataset",
            "val_dataset",
            "test_dataset",
            "train_ray_generator",
            "val_ray_generator",
            "test_ray_generator",
            "train_pixel_sampler",
            "val_pixel_sampler",
            "test_pixel_sampler",
        ):
            obj = getattr(self, name, None)
            if obj is not None and hasattr(obj, "to"):
                try:
                    obj.to(self.device)
                except Exception:
                    # Don't fail the whole move if a child can't be moved; it's
                    # better to let model/device mismatches surface elsewhere.
                    pass

        return self

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch
