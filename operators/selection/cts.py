from typing import TYPE_CHECKING, Sequence

import torch

from term import Term
from util import l2_distance

from .ts import TS

if TYPE_CHECKING:
    from gp import GPSolver        
    
class CTS(TS):
    ''' Competent tournament Selection  '''
    def __init__(self, name: str = "CTS", **kwargs):
        super().__init__(name, **kwargs)    

    def select(self, solver: 'GPSolver', population: Sequence[Term], selection_size: int) -> Sequence[Term]:
        half_size = selection_size // 2
        half_parents = super().select(solver, population, half_size + (selection_size % 2)) 
        half_parents_sems = solver.eval(half_parents, return_outputs="tensor").outputs
        half_parents_dist = l2_distance(half_parents_sems, solver.target)
        children = []
        for i in range(half_size):
            first_parent = half_parents[i]
            first_sem = half_parents_sems[i]
            first_target_dist = half_parents_dist[i]
            # find second parent
            candidiates = solver.rnd.choice(population, size = self.tournament_size)
            candidate_sem = solver.eval(candidiates, return_outputs="tensor").outputs

            candidate_target_dist = l2_distance(candidate_sem, solver.target)
            candidate_parent_dist = l2_distance(candidate_sem, first_sem)
            del candidate_sem
            cand_scores = candidate_target_dist / candidate_parent_dist * (1.0 + torch.abs(first_target_dist - candidate_target_dist))
            cand_scores.nan_to_num_(nan=torch.inf)
            best_cand_id = torch.argmin(cand_scores).item()
            best_candidate = candidiates[best_cand_id]

            children.append(first_parent)
            children.append(best_candidate)

        del half_parents_sems

        if selection_size % 2 == 1:
            children.append(half_parents[-1])
        return children