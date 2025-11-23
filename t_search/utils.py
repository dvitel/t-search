''' Set of useful objects and functions '''
from time import perf_counter
from typing import Callable, Sequence
import numpy as np
from pyparsing import Literal
import torch

GLOBAL_RNG = np.random.default_rng() 

def add_metric(metrics: dict, **kwargs):
    for key, value in kwargs.items():
        if key not in metrics:
            metrics[key] = value            
        elif isinstance(metrics[key], list):
            metrics[key].extend(value)
        else:
            metrics[key] = metrics[key] + value   

def timed(fn: Callable) -> Callable:
    """Decorator to time function execution"""

    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        try:
            result = fn(*args, **kwargs)
        finally:
            elapsed_time = round((perf_counter() - start_time) * 1000)
        return result, elapsed_time

    return wrapper

def stack_rows(tensors: Sequence[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    if len(tensors) == 0:
        return torch.empty((0, target.shape[0]), dtype=target.dtype, device=target.device)
    if tensors[0].ndim <= 1:
        res = torch.empty((len(tensors), target.shape[0]), dtype=tensors[0].dtype, device=tensors[0].device)
        for i, ti in enumerate(tensors):
            res[i] = ti # assuming broadcastable
        return res  
    if tensors[0].ndim == 2:
        sz = (sum(t.shape[0] for t in tensors), target.shape[0])
        res = torch.empty(sz, dtype=tensors[0].dtype, device=tensors[0].device)
        cur_start = 0
        for ti in tensors:
            res[cur_start:cur_start + ti.shape[0]] = ti
            cur_start += ti.shape[0]
        return res  
    raise ValueError(f"Unsupported tensor shape: {tensors[0].shape}")

GPSolverStatus = Literal["INIT", "MAX_GEN", "MAX_EVAL", "MAX_ROOT_EVAL", "SOLVED"]

class EvSearchTermination(Exception):
    """Reaching maximum of evals, gens, ops etc"""

    def __init__(self, status: GPSolverStatus, *args):
        super().__init__(*args)
        self.status = status