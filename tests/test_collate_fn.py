"""Regression tests for collate_fn label padding."""

import torch

from cryovit.datamodules.utils import collate_fn
from cryovit.types import TomogramData


def _make_tomo(d: int, c: int = 1, h: int = 4, w: int = 4, name: str = "t"):
    data = torch.zeros(c, d, h, w)
    data[:, :, 0, 0] = 1.0  # marker in features
    label = torch.zeros(d, h, w, dtype=torch.bool)
    label[0, 0, 0] = True
    return TomogramData(
        sample="s",
        tomo_name=name,
        split_id=None,
        data=data,
        label=label,
        aux_data={},
    )


def test_collate_pads_label_not_data():
    """Unequal depths must pad labels with -1 and features with 0."""
    short = _make_tomo(d=2, name="short")
    tall = _make_tomo(d=5, name="tall")
    batch = collate_fn([short, tall])

    assert batch.labels.shape == (2, 5, 4, 4)
    assert batch.tomo_batch.shape[1] == 5

    # short tomogram: real slices keep the label marker; pad region is -1
    assert batch.labels[0, 0, 0, 0] == 1
    assert batch.labels[0, 1, 1, 1] == 0
    assert torch.all(batch.labels[0, 2:] == -1)

    # tall tomogram is unpadded
    assert torch.all(batch.labels[1, :5] != -1)

    # feature padding must be 0, not -1 (old bug padded data with -1)
    assert torch.all(batch.tomo_batch[0, 2:, :, 1:, 1:] == 0)
    assert torch.all(batch.tomo_batch[0, 0, :, 0, 0] == 1.0)


def test_collate_equal_depth_no_pad():
    a = _make_tomo(d=3, name="a")
    b = _make_tomo(d=3, name="b")
    batch = collate_fn([a, b])
    assert batch.labels.shape == (2, 3, 4, 4)
    assert not torch.any(batch.labels == -1)


def test_collate_channels_preserved():
    short = _make_tomo(d=1, c=2, name="short")
    tall = _make_tomo(d=4, c=2, name="tall")
    batch = collate_fn([short, tall])
    assert batch.tomo_batch.shape == (2, 4, 2, 4, 4)  # B, D, C, H, W
    assert torch.all(batch.labels[0, 1:] == -1)
