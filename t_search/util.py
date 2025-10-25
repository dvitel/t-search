from typing import TYPE_CHECKING, Sequence

import torch

from t_search.term import Term

if TYPE_CHECKING:
    from t_search.solver import GPSolver


def stack_rows(tensors: Sequence[torch.Tensor], target_size: int | None = None) -> torch.Tensor:
    if target_size is None:
        target_size = max(0 if len(ti.shape) == 0 else ti.shape[0] for ti in tensors)
    sz = (len(tensors), ) if target_size == 0 else (len(tensors), target_size)
    res = torch.empty(sz, dtype=tensors[0].dtype, device=tensors[0].device)
    for i, ti in enumerate(tensors):
        res[i] = ti # assuming broadcastable
    return res  

def stack_rows_2d(tensors: Sequence[torch.Tensor], target_size: int) -> torch.Tensor:
    sz = (sum(t.shape[0] for t in tensors), target_size)
    res = torch.empty(sz, dtype=tensors[0].dtype, device=tensors[0].device)
    cur_start = 0
    for ti in tensors:
        res[cur_start:cur_start + ti.shape[0]] = ti
        cur_start += ti.shape[0]
    return res  

def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    el_dist = (a - b) ** 2
    el_dist.nan_to_num_(nan=torch.inf)
    return torch.sqrt(torch.sum(el_dist, dim=-1))