''' Selection operators '''

from typing import TYPE_CHECKING, Sequence

import torch

from term import Term
from util import Operator, l2_distance, stack_rows

if TYPE_CHECKING:
    from gp import GPSolver  # Import only for type checking


class Selection(Operator):
    ''' Base class for selection operators '''

    def __init__(self, name: str = "randSel", replace: bool = True):
        super().__init__(name)
        self.replace = replace
    
    def exec(self, solver: 'GPSolver', population):
        return self.select(solver, population, solver.pop_size)
    
    def select(self, solver: 'GPSolver', population: Sequence[Term], selection_size: int) -> Sequence[Term]:
        children = solver.rnd.choice(population, selection_size, replace=self.replace).tolist()
        return children

class TournamentSelection(Selection):
    ''' Tournament selection operator '''
    
    def __init__(self, name: str = "tournament", tournament_size: int = 7):
        super().__init__(name)
        self.tournament_size = tournament_size

    def select(self, solver: 'GPSolver', population, selection_size: int) -> Sequence[Term]:
        ''' Fitness is 1d tensor of fitness selected for tournament '''
        fitness = solver.eval(population, return_fitness="tensor").fitness
        selected_ids = torch.randint(fitness.shape[0], (selection_size, self.tournament_size), dtype=torch.int, device=fitness.device,
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

class Elitism(Operator): 
    ''' Passes through the population but stores elite terms'''

    def __init__(self, name: str = "elitism", size: int = 10):
        super().__init__(name)
        self.elite_size = size

    def __call__(self, solver: 'GPSolver', population: Sequence[Term], next_ops: list['Operator'] = []):
        elite: list[Term] = []
        fitness = solver.eval(population, return_fitness="tensor").fitness
        sorted_ids = torch.argsort(fitness, dim=0)
        elite_ids = sorted_ids[:self.elite_size].tolist()
        del fitness, sorted_ids
        elite = [population[i] for i in elite_ids]
        # bad_ids = sorted_ids[-self.elite_size:].tolist()
        # passed_population = [] 
        # bad_id_set = set(bad_ids)
        # for i, term in enumerate(population):
        #     if i not in bad_id_set:
        #         passed_population.append(term)
        children = self.call_next(solver, population, next_ops)
        # children are not evaluated yet - add elite anyway
        children.extend(elite) 
        return children
    
class Finite(Operator):
    ''' Selects only children that have finite or unknown outputs 
        Resorts back to full population if all outputs are infinie or nan
    '''
    def __init__(self, name: str = "finite"):
        super().__init__(name)

    def exec(self, solver: 'GPSolver', population: list[Term]):
        children = [ch for ch in population if ch not in solver.invalid_term_outputs]
        if len(children) == 0:
            print("WARN: all population has nans or infs")
            return population
        return children
    
class STS(TournamentSelection):
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


class CTS(TournamentSelection):
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
