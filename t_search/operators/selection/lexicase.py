from typing import TYPE_CHECKING, Sequence
import torch

from t_search.term import Term
from .base import Selection

if TYPE_CHECKING:
    from t_search.solver import GPSolver

class LexicaseSelection(Selection):
    ''' Lexicase selection operator '''
    
    def __init__(self, name: str = "lexicase", nan_error = torch.inf,):
        super().__init__(name)
        self.nan_error = nan_error

    def select(self, solver: 'GPSolver', population, selection_size: int) -> Sequence[Term]:
        outputs = solver.eval(population, return_outputs="tensor").outputs

        nan_interactions = torch.abs(outputs - solver.target)

        interactions = torch.nan_to_num(nan_interactions, nan=self.nan_error)
        del nan_interactions
        
        selected_ids = torch.zeros(selection_size, dtype=torch.int, device=outputs.device)

        for pos_i in range(solver.pop_size):
            shuffled_test_ids = torch.randperm(interactions.shape[-1], device=interactions.device,
                                                generator=solver.torch_gen)
            candidate_ids = torch.arange(interactions.shape[0], device=interactions.device) # all candidates
            for test_id in shuffled_test_ids:
                test_min_diff = torch.min(interactions[candidate_ids, test_id])
                candidate_id_ids, = torch.where(interactions[candidate_ids, test_id] == test_min_diff)
                candidate_ids = candidate_ids[candidate_id_ids]
                if len(candidate_ids) == 1:
                    break
            if len(candidate_ids) == 1:
                selected_ids[pos_i] = candidate_ids[0]
                continue
            best_id_id = torch.randint(len(candidate_ids), (1,), device=interactions.device,
                                        generator=solver.torch_gen)
            selected_ids[pos_i] = candidate_ids[best_id_id]
        del interactions, shuffled_test_ids
        del outputs
        children = [population[term_id] for term_id in selected_ids.tolist()]
        return children