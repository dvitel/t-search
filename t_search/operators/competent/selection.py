from typing import Sequence

import torch

from t_search.evaluators.semantics import Semantics
from t_search.operators.classic.ts import TS
from t_search.syntax import Term
from t_search.evaluators.fitness import l2

class CompetentSelection(TS):
    ''' Competent tournament Selection  '''
    def __init__(self, *, 
                  semantics: Semantics,
                  target: torch.Tensor,
                  **kwargs):
        super().__init__(**kwargs)
        self.semantics = semantics
        self.target = target

    def __call__(self, population: Sequence[Term]) -> Sequence[Term]:
        half_size = self.selection_size // 2
        half_parents = super().__call__(population, size=half_size + (self.selection_size % 2)) 
        half_parents_sems = self.semantics.get_outputs(half_parents, return_type="tensor")
        half_parents_dist = l2(half_parents_sems, self.target)
        children = []
        for i in range(half_size):
            first_parent = half_parents[i]
            first_sem = half_parents_sems[i]
            first_target_dist = half_parents_dist[i]
            # find second parent
            candidiates = self.rnd.choice(population, size = self.tournament_size)
            candidate_sem = self.semantics.get_outputs(candidiates, return_type="tensor")

            candidate_target_dist = l2(candidate_sem, self.target)
            candidate_parent_dist = l2(candidate_sem, first_sem)
            del candidate_sem
            cand_scores = candidate_target_dist / candidate_parent_dist * (1.0 + torch.abs(first_target_dist - candidate_target_dist))
            cand_scores.nan_to_num_(nan=torch.inf)
            best_cand_id = torch.argmin(cand_scores).item()
            best_candidate = candidiates[best_cand_id]

            children.append(first_parent)
            children.append(best_candidate)

        del half_parents_sems

        if self.selection_size % 2 == 1:
            children.append(half_parents[-1])
        return children