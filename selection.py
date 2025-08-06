''' Selection operators '''

from typing import TYPE_CHECKING

import torch

from util import stack_rows

if TYPE_CHECKING:
    from gp import GPSolver  # Import only for type checking


class Selection:
    ''' Base class for selection operators '''
    
    def __init__(self, name: str = "selection"):
        self.name = name

    def __call__(self, solver: 'GPSolver', population, num_select: int):
        children = solver.rnd.choice(population, num_select, replace=False)
        return children
    

class TournamentSelection(Selection):
    ''' Tournament selection operator '''
    
    def __init__(self, name: str = "tournament", tournament_size: int = 7):
        super().__init__(name)
        self.tournament_size = tournament_size

    def __call__(self, solver: 'GPSolver', population, num_select: int):
        ''' Fitness is 1d tensor of fitness selected for tournament '''
        fitness_list = [solver.term_fitness[term] for term in population]
        fitness = torch.stack(fitness_list, dim=0)
        selected_ids = torch.randint(fitness.shape[0], (num_select, self.tournament_size), dtype=torch.int, device=fitness.device,
                                    generator=solver.torch_gen)
        selected_fitnesses = fitness[selected_ids]
        best_id_id = torch.argmin(selected_fitnesses, dim=-1)
        best_ids = torch.gather(selected_ids, dim=-1, index = best_id_id.unsqueeze(-1)).squeeze(-1)
        del selected_ids, selected_fitnesses, best_id_id
        del fitness
        children = [population[best_id] for best_id in best_ids.tolist()]
        return children
    
class LexicaseSelection(Selection):
    ''' Lexicase selection operator '''
    
    def __init__(self, name: str = "lexicase", nan_error = torch.inf,):
        super().__init__(name)
        self.nan_error = nan_error

    def __call__(self, solver: 'GPSolver', population, num_select: int):
        outputs_list = [solver.get_cached_output(term) for term in population]
        outputs = stack_rows(outputs_list, target_size=solver.target.shape[0])

        nan_interactions = torch.abs(outputs - solver.target)

        interactions = torch.nan_to_num(nan_interactions, nan=self.nan_error)
        del nan_interactions
        
        selected_ids = torch.zeros(num_select, dtype=torch.int, device=outputs.device)

        for pos_i in range(num_select):
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

