from torch import Tensor

from nerfstudio.fields.nerfacto_field import NerfactoField


class NadirNerfField(NerfactoField):

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
    ) -> None:
        super().__init__(aabb=aabb, num_images=num_images)
