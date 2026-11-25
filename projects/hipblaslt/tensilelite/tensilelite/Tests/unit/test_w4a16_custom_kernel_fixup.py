# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The Stream-K fixup of the shipped gfx115x W4A16 custom kernels.

The owner of a split tile adds every other workgroup's partial into its
accumulators, one batch at a time, waiting for each batch's loads before the
next. A kernel emitted with the fixup batches capped at the global write's
NumElementsPerBatchStore (2, 4 or 8 elements) adds a partial in 6 to 40
serialized load round trips: about 3 us per partial instead of about 1 on
gfx1151. The kernels add a partial in as few batches as their free VGPRs allow:
one or two, three for MT64x224x64."""
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_CUSTOM_KERNEL_DIR = Path(__file__).parents[2] / "CustomKernels"
_MAX_FIXUP_BATCHES = 3


def _w4a16_stream_k_kernels():
    return sorted(_CUSTOM_KERNEL_DIR.glob("RuntimeGroup_Cijk_Alik_Bljk_I4*_UserArgs_MT*_SK*_gfx115*.s"))


@pytest.mark.skipif(
    not _CUSTOM_KERNEL_DIR.is_dir(),
    reason=f"CustomKernels directory not found: {_CUSTOM_KERNEL_DIR}",
)
def test_w4a16_stream_k_kernels_add_a_partial_in_few_batches():
    kernels = _w4a16_stream_k_kernels()
    assert kernels, "no gfx115x W4A16 Stream-K custom kernels found"
    batches = {k.name: len(re.findall(r"^/\* Fixup Batch #", k.read_text(), re.M)) for k in kernels}
    # Guards against passing vacuously if the batch headers ever change.
    assert all(n >= 1 for n in batches.values()), \
        [name for name, n in batches.items() if n < 1][:3]
    over = {name: n for name, n in batches.items() if n > _MAX_FIXUP_BATCHES}
    assert not over, f"{len(over)} kernels add a partial in more than {_MAX_FIXUP_BATCHES} batches: " \
        f"{dict(list(over.items())[:4])}"
