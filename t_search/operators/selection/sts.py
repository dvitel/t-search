

from typing import TYPE_CHECKING, Sequence

import torch

from syntax import Term
from .ts import TS

if TYPE_CHECKING:
    from t_search.solver import GPSolver        
    
class STS(TS):
    ''' Semantic Tournament Selection Galván-López et al. (2013)
        First parent with TS, second is the best parent != to first one or first one. 
        "Using Semantics in Selection Mechanism in Genetic Programming: a Simple Methods for Promoting Semantic Diversity"
    '''
    def __init__(self, name: str = "STS", *, rtol = 1e-5, atol=1e-4, **kwargs):
        super().__init__(name, **kwargs)    
        self.rtol = rtol
        self.atol = atol

    def select(self, solver: 'GPSolver', population: Sequence[Term], selection_size: int) -> Sequence[Term]:
        half_size = selection_size // 2
        half_parents = super().select(solver, population, half_size + (selection_size % 2)) 
        half_parents_sems = solver.eval(half_parents, return_outputs="tensor").outputs
        children = []
        for i in range(half_size):
            first_parent = half_parents[i]
            first_sem = half_parents_sems[i]
            # find second parent
            candidiates = solver.rnd.choice(population, size = self.tournament_size)
            candidate_sem = solver.eval(candidiates, return_outputs="tensor").outputs
            # cand_dist = torch.sqrt((candidate_sem - first_sem) ** 2)
            mask = torch.isclose(candidate_sem, first_sem, rtol=self.rtol, atol=self.atol).all(dim=-1)
            del candidate_sem
            filter_ids, = torch.where(~mask)
            filtered_candidates = [candidiates[i] for i in filter_ids.tolist()]
            c_fitness = solver.eval(filtered_candidates, return_fitness="tensor").fitness
            best_id = torch.argmin(c_fitness).item()
            best_parent = candidiates[best_id]
            del c_fitness
            children.append(first_parent)
            children.append(best_parent)

        del half_parents_sems

        if selection_size % 2 == 1:
            children.append(half_parents[-1])
        return children