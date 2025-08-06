from typing import Sequence

import torch


def stack_rows(tensors: Sequence[torch.Tensor], target_size: int | None = None) -> torch.Tensor:
    if target_size is None:
        target_size = max(0 if len(ti.shape) == 0 else ti.shape[0] for ti in tensors)
    sz = (len(tensors), ) if target_size == 0 else (len(tensors), target_size)
    res = torch.empty(sz, dtype=tensors[0].dtype, device=tensors[0].device)
    for i, ti in enumerate(tensors):
        res[i] = ti # assuming broadcastable
    return res  