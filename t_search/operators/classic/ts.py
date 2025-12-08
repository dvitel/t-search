

from typing import Sequence

import torch

from t_search.evaluators.evaluator import Evaluator
from t_search.operators.selection import Selection
from t_search.syntax import Term

class TS(Selection):
    ''' Tournament selection operator '''
    
    def __init__(self, *,
                 evaluator: Evaluator,
                 torch_gen: torch.Generator,
                 tournament_size: int = 7, 
                 **kwargs):
        super().__init__(**kwargs)
        self.tournament_size = tournament_size
        self.evaluator = evaluator
        self.torch_gen = torch_gen

    def __call__(self, population: Sequence[Term]) -> Sequence[Term]:
        ''' Fitness is 1d tensor of fitness selected for tournament '''
        fitness = self.evaluator.eval(population, return_fitness="tensor").fitness
        selected_ids = torch.randint(fitness.shape[0], 
                                     (self.selection_size, self.tournament_size), 
                                     dtype=torch.int, device=fitness.device,
                                     generator=self.torch_gen)
        selected_fitnesses = fitness[selected_ids]
        best_id_id = torch.argmin(selected_fitnesses, dim=-1)
        best_ids = torch.gather(selected_ids, dim=-1, index = best_id_id.unsqueeze(-1)).squeeze(-1)
        del selected_ids, selected_fitnesses, best_id_id
        del fitness
        children = [population[best_id] for best_id in best_ids.tolist()]
        return children
