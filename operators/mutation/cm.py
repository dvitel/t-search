

from dataclasses import dataclass, field
from typing import Sequence, TYPE_CHECKING

import torch

from initialization import UpToDepth
from ..base import TermsListener
from .base import PositionMutation
from term import Term, TermPos
from term_spatial import TermVectorStorage
from torch_alg_inv import DesiredSemantics, alg_inv, get_desired_semantics, invert
if TYPE_CHECKING:
    from gp import GPSolver

def get_best_semantics(desired: DesiredSemantics, undesired: list[DesiredSemantics], all_semantics: torch.Tensor,):
    assert len(desired) > 0

    if any(d is None for d in desired): # unsat desired at position 
        return None

    # if all(len(d) == 0 for d in desired): # any term will work - shou
    #     return None 

    forbidden_mask = torch.zeros((all_semantics.shape[1],), dtype=torch.bool, device=all_semantics.device)

    for forbidden in undesired:

        if any(d is None for d in forbidden):
            continue # unsat undesired - skip
        
        forbidden_close_mask = torch.ones((all_semantics.shape[1],), dtype=torch.bool, device=all_semantics.device)

        for test_id, forbit_values in enumerate(forbidden):
            if len(forbit_values) == 0:
                continue 
            sem_values = all_semantics[:, test_id].unsqueeze(-1) # (num_terms, 1)
            forbidden_tensor = torch.tensor(list(forbit_values), dtype=all_semantics.dtype, device=all_semantics.device)
            diffs = torch.abs(sem_values - forbidden_tensor.unsqueeze(0)) # (num_terms, num_forbidden)
            close_mask = torch.any(diffs < 1e-5, dim=1) # (num_terms,)
            forbidden_close_mask &= close_mask
            del forbidden_tensor
            if not torch.any(forbidden_close_mask):
                break

        forbidden_mask |= forbidden_close_mask

    # test_ids = [i for i, d in enumerate(desired) if len(d) > 0]
    # selected_semantics = all_semantics[:, test_ids]
    sem_score = torch.zeros((all_semantics.shape[0],), dtype=all_semantics.dtype, device=all_semantics.device)
    for test_id, allowed_values in enumerate(desired):
        if len(allowed_values) == 0:
            continue 
        sem_values = all_semantics[:, test_id].unsqueeze(-1) # (num_terms, 1)
        allowed_tensor = torch.tensor(list(forbit_values), dtype=all_semantics.dtype, device=all_semantics.device)
        diffs = torch.abs(sem_values - allowed_tensor.unsqueeze(0)) # (num_terms, num_allowed)
        sem_score += torch.min(diffs, dim=1).values # (num_terms,) 

    sem_score[forbidden_mask] = torch.inf

    best_sem_id = torch.argmin(sem_score).item()

    if sem_score[best_sem_id] == torch.inf:
        return None

    return best_sem_id

@dataclass 
class InversionCache: 
    term_semantics: dict[Term, DesiredSemantics] = field(default_factory=dict)
    term_subtree_semantics: dict[Term, dict[tuple[Term, int], tuple[DesiredSemantics, list[DesiredSemantics]]]] = field(default_factory=dict)

class CM(PositionMutation, TermsListener):
    ''' Competent Mutation from Dr. Kraviec and Pawlak
        Parent program is lineary combined with random term 
    '''
    def __init__(self, name = "CM", *, 
                    index: TermVectorStorage, 
                    inv_cache: InversionCache,
                    index_init_depth: int | None = None, 
                    dynamic_index: bool = False,
                    index_max_size: int = 1e10,
                    op_invs = alg_inv,
                    **kwargs):
        super().__init__(name, **kwargs)
        self.index = index # used as library of semantics 
        self.inv_cache = inv_cache
        self.index_init_depth = index_init_depth # if None, dynamic library - uses any available term. 
        self.dynamic_index = dynamic_index
        self.index_max_size = index_max_size
        self.desired_target: DesiredSemantics | None = None
        self.op_invs = op_invs
        self.desired_at_pos = {} # temp cache

    def op_init(self, solver):
        ''' Initializes desired combinatorial semantics and Library of programs '''
        self.desired_target = get_desired_semantics(solver.target)
        if self.index_init_depth is not None and self.index.len_sem() == 0: 
            init_op = UpToDepth(self.index_init_depth, force_pop_size=False)
            lib_terms = init_op(solver, pop_size=self.index_max_size)
            semantics = solver.eval(lib_terms, return_outputs="tensor").outputs
            self.index.insert(lib_terms, semantics) 
            del semantics
        pass 

    def register_terms(self, solver, terms, semantics):
        if self.dynamic_index and self.index.len_terms() < self.index_max_size:
            self.index.insert(terms, semantics)
        return []

    def mutate_position(self, solver: 'GPSolver', term: Term, position: TermPos) -> Term | None:
        
        if (position.term, position.occur) not in self.desired_at_pos:
            return None
        
        desired, undesired = self.desired_at_pos[(position.term, position.occur)]

        all_semantics = self.index.get_semantics()

        best_sem_id = get_best_semantics(desired, undesired, all_semantics)

        if best_sem_id is None:
            return None
        
        best_term = self.index.get_repr_term(best_sem_id)
        
        mutated_term = solver.replace_position(term, position, best_term)

        return mutated_term

    
    def mutate_term(self, solver: 'GPSolver', term: Term, parents: Sequence[Term], children: Sequence[Term]) -> Term | None:

        term_sem, *_ = solver.eval(term, return_outputs="list").outputs
        if term not in self.inv_cache.term_semantics:
            self.inv_cache.term_semantics[term] = get_desired_semantics(term_sem)

        self.desired_at_pos = invert(term, self.desired_target, [self.inv_cache.term_semantics[term]], 
                                     lambda args: solver.eval(args, return_outputs="list").outputs, 
                                     self.inv_cache.term_semantics, self.op_invs)
        
        child = super().mutate_term(solver, term, parents, children)

        del self.desired_at_pos

        return child