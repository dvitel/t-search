from typing import Sequence

import torch

from t_search.evaluators.evaluator import Evaluator
from t_search.syntax import Term
from t_search.evaluators.fitness import l2

from .ts import TS

class CTS(TS):
    ''' Competent tournament Selection  '''
    def __init__(self, *, 
                  evaluator: Evaluator,
                  target: torch.Tensor,
                  **kwargs):
        super().__init__(**kwargs)
        self.evaluator = evaluator
        self.target = target

    def select(self, population: Sequence[Term], selection_size: int) -> Sequence[Term]:
        half_size = selection_size // 2
        half_parents = super().select(population, half_size + (selection_size % 2)) 
        half_parents_sems = self.evaluator.eval(half_parents, return_outputs="tensor").outputs
        half_parents_dist = l2(half_parents_sems, self.target)
        children = []
        for i in range(half_size):
            first_parent = half_parents[i]
            first_sem = half_parents_sems[i]
            first_target_dist = half_parents_dist[i]
            # find second parent
            candidiates = self.rnd.choice(population, size = self.tournament_size)
            candidate_sem = self.evaluator.eval(candidiates, return_outputs="tensor").outputs

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

        if selection_size % 2 == 1:
            children.append(half_parents[-1])
        return children