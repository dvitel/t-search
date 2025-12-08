from typing import Sequence
import torch

from t_search.evaluators.evaluator import Evaluator
from t_search.operators.selection import Selection
from t_search.syntax import Term

class Lexicase(Selection):
    ''' Lexicase selection operator '''
    
    def __init__(self, 
                 evaluator: Evaluator,
                 target: torch.Tensor,
                 torch_gen: torch.Generator,
                 nan_error = torch.inf,
                 **kwargs):
        super().__init__(**kwargs)
        self.nan_error = nan_error
        self.evaluator = evaluator
        self.target = target
        self.torch_gen = torch_gen

    def __call__(self, population) -> Sequence[Term]:
        outputs = self.evaluator.eval(population, return_outputs="tensor").outputs

        nan_interactions = torch.abs(outputs - self.target)

        interactions = torch.nan_to_num(nan_interactions, nan=self.nan_error)
        del nan_interactions
        
        selected_ids = torch.zeros(self.selection_size, dtype=torch.int, device=outputs.device)

        for pos_i in range(self.selection_size):
            shuffled_test_ids = torch.randperm(interactions.shape[-1], device=interactions.device,
                                                generator=self.torch_gen)
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
                                        generator=self.torch_gen)
            selected_ids[pos_i] = candidate_ids[best_id_id]
        del interactions, shuffled_test_ids
        del outputs
        children = [population[term_id] for term_id in selected_ids.tolist()]
        return children