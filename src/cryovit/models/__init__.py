"""Implementations of deep-learning automated segmentation models for CryoViT."""

from cryovit.models.base_model import BaseModel
from cryovit.models.cryovit import CryoVIT

# sam2 is optional at import time (DINO features / infer work without it).
try:
    from cryovit.models.sam2 import SAM2, create_sam_model_from_weights
except ImportError:  # pragma: no cover
    SAM2 = None

    def create_sam_model_from_weights(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "sam2 is not installed; install `sam-2` to use SAM features/training."
        )


from cryovit.models.unet3d import UNet3D

__all__ = [
    "BaseModel",
    "CryoVIT",
    "UNet3D",
    "SAM2",
    "create_sam_model_from_weights",
]
