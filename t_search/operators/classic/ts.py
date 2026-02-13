

from typing import Sequence

import torch

from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.operators.selection import Selection
from t_search.syntax import Term

class TS(Selection):
    ''' Tournament selection operator '''
    
    def __init__(self, *,
                 fitness: Fitness,
                 torch_gen: torch.Generator,
                 tournament_size: int = 7, 
                 add_pop: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.tournament_size = tournament_size
        self.fitness = fitness
        self.torch_gen = torch_gen
        self.add_pop = add_pop

    def __call__(self, population: Sequence[Term], size: int | None = None) -> Sequence[Term]:
        ''' Fitness is 1d tensor of fitness selected for tournament '''
        fitness = self.fitness.get_fitness(population, return_type="tensor")
        selected_ids = torch.randint(fitness.shape[0], 
                                     (size or self.selection_size, self.tournament_size), 
                                     dtype=torch.long, device=fitness.device,
                                     generator=self.torch_gen)
        selected_fitnesses = fitness[selected_ids]
        best_id_id = torch.argmin(selected_fitnesses, dim=-1)
        best_ids = torch.gather(selected_ids, dim=-1, index = best_id_id.unsqueeze(-1)).squeeze(-1)
        del selected_ids, selected_fitnesses, best_id_id
        del fitness
        children = [population[best_id] for best_id in best_ids.tolist()]
        if self.add_pop: # guarantee that parents appear at least once
            present_terms = set(children)
            for parent in population:
                if parent not in present_terms:
                    children.append(parent)
                    present_terms.add(parent)
        return children
