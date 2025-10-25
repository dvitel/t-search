from typing import TYPE_CHECKING, Sequence

import torch

from term import Term

if TYPE_CHECKING:
    from gp import GPSolver


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

class Operator:
    def __init__(self, name: str):
        self.name = name 
        self.metrics = {}

    def reset_metrics(self):
        self.metrics = {}
    
    def exec(self, solver: 'GPSolver', population: Sequence[Term]) -> Sequence[Term]:
        ''' Executes only this operator and update existing metrics state '''
        return population
    
    def call_next(solver: 'GPSolver', population: Sequence[Term], next_ops: list['Operator'] = []):
        if len(next_ops) > 0:
            next_op, *rest_ops = next_ops
            children = next_op(solver, children, rest_ops)        
            return children
        return population

    def __call__(self, solver: 'GPSolver', population: Sequence[Term], next_ops: list['Operator'] = []):
        ''' Executes operator in the chain. New metrics are stored in self.metrics '''
        self.reset_metrics()
        children = self.exec(solver, population)
        children = self.call_next(solver, children, next_ops)
        return children


class OperatorInitMixin: 
    def op_init(self, solver: 'GPSolver'):
        pass

class TermsListener: 
    ''' Interface to listen for new terms appearing during the eval. 
        
    '''
    def register_terms(self, solver: 'GPSolver', terms: list[Term], semantics: torch.Tensor) -> list[Term]: 
        pass 