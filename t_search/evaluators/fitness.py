from functools import partial
from typing import Callable, Literal, Optional
import torch

from t_search.base import ServiceBase
from t_search.evaluators.semantics import Semantics
from t_search.syntax.syntax import Syntax
from t_search.syntax.term import Term
from t_search.utils import EvSearchTermination


def mse(vector, target):
    return (vector - target) ** 2


def nmse(vector, target, target_variance: torch.Tensor | None = None) -> torch.Tensor:
    """we follow R^2 normalization: NMSE = 1 - R^2"""        

    res = mse(vector, target)
    variance_mask_clamped = torch.clamp(target_variance, min=1e-08)   
    res /= variance_mask_clamped

    return res

def l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    el_dist = (a - b) ** 2
    el_dist.nan_to_num_(nan=torch.inf)
    return torch.sqrt(torch.sum(el_dist, dim=-1))

class Fitness(ServiceBase): 

    def __init__(self, *,
                 syntax: Syntax,
                 target: torch.Tensor,
                 target_variance: torch.Tensor,
                 fitness_atol: float = 1e-6,
                 fitness_err: float = 1e-6,
                 debug: bool = False):
        self.target = target
        self.target_variance = target_variance
        self.syntax = syntax
        self.fitness: dict[Term, torch.Tensor] = {}
        self.best_term: Optional[Term] = None
        self.best_term_fitness: Optional[torch.Tensor] = None
        self.best_term_depth: Optional[int] = None
        self.best_term_size: Optional[int] = None
        self.best_term_outputs: Optional[torch.Tensor] = None
        self.fitness_atol = fitness_atol
        self.fitness_err = fitness_err    

        self.debug = debug

    def pick_best_around(self, target_fitness_id: int, terms: list[Term], outputs: torch.Tensor, fitness: torch.Tensor) -> int:
        if len(terms) <= 1:
            return target_fitness_id
        target_fitness = fitness[target_fitness_id]
        close_ids = torch.where(torch.isclose(fitness, target_fitness, atol=self.fitness_err, rtol=0))[0]
        if len(close_ids) > 1:
            close_ids_size_depths = [(self.syntax.get_size(terms[i]), self.syntax.get_depth(terms[i]), i) for i in close_ids.tolist()]
            # take minimal size, then depth, then i 
            min_val = close_ids_size_depths[0]
            for idx, val in enumerate(close_ids_size_depths):
                if val < min_val:
                    min_val = val
            target_fitness_id = min_val[2]        
        return target_fitness_id

    def set_best_term(self, terms: list[Term], outputs: torch.Tensor, fitness: torch.Tensor):
        if len(outputs) == 0:
            return
        new_term = None 
        best_new_fitness, best_new_id = torch.min(fitness, dim=0)
        if self.best_term is None:
            best_new_id = self.pick_best_around(best_new_id.item(), terms, outputs, fitness)
            new_term = terms[best_new_id]
        # if torch.isclose(best_new_fitness, self.best_term_fitness, atol=self.fitness_err, rtol=0):
        #     # very similar by fitness - pick best by (size, depth)
        #     best_new_depth = self.syntax.get_depth(new_term)
        #     best_new_size = self.syntax.get_size(new_term)
        #     if 
        elif torch.isclose(best_new_fitness, self.best_term_fitness, atol=self.fitness_err, rtol=0):
            best_new_id = self.pick_best_around(best_new_id.item(), terms, outputs, fitness)
            new_term = terms[best_new_id]
            new_size = self.syntax.get_size(new_term)
            new_depth = self.syntax.get_depth(new_term)
            if not ((new_size, new_depth) < (self.best_term_size, self.best_term_depth)):
                new_term = None
        elif best_new_fitness < self.best_term_fitness:
            best_new_id = self.pick_best_around(best_new_id.item(), terms, outputs, fitness)
            new_term = terms[best_new_id]

        if new_term is not None:
            self.best_term = new_term
            self.best_term_fitness = best_new_fitness
            self.best_term_outputs = outputs[best_new_id]
            self.best_term_depth = self.syntax.get_depth(new_term)
            self.best_term_size = self.syntax.get_size(new_term)
            if self.debug:
                print(f"New best {self.best_term} fitness={self.best_term_fitness.item():.6e}, size={self.best_term_size}, depth={self.best_term_depth}")
            if self.best_term_fitness < self.fitness_atol:
                raise (EvSearchTermination("SOLVED"))       

        return

    def get_missing(self, terms: list[Term] | Term) -> list[Term]:
        if isinstance(terms, Term):
            terms = [terms]
        missing_terms = [t for t in terms if t not in self.fitness]
        return missing_terms

    def get_fitness(self, terms: list[Term] | Term, return_type: Literal["list", "tensor"] = "list") -> list[torch.Tensor] | torch.Tensor | None:
        if isinstance(terms, Term):
            return self.fitness.get(terms, None)
        selected_fitness = []
        for t in terms:
            t_fitness = self.fitness.get(t, None)
            if t_fitness is None:
                raise ValueError(f"Term {t} fitness not found")
            selected_fitness.append(t_fitness)
        if return_type == "tensor":
            return torch.stack(selected_fitness)
        return selected_fitness

    def set_fitness(self, valid_terms: list[Term], valid_semantics: torch.Tensor) -> None:
        fitness_per_test = nmse(valid_semantics, target=self.target.unsqueeze(0), target_variance=self.target_variance)
        fitness_per_test.nan_to_num_(nan=torch.inf)
        fitness = torch.mean(fitness_per_test, dim=-1)
        for term, fit in zip(valid_terms, fitness):
            self.fitness[term] = fit.clone()
        self.set_best_term(valid_terms, valid_semantics, fitness)
        del fitness
        return
    
    def copy_fitness(self, from_term: Term, to_term: Term) -> None:
        self.fitness[to_term] = self.fitness[from_term]
        return

    def get_finalizer(self):

        def finalize():
            for fit in self.fitness.values():
                del fit
            self.fitness.clear()
        return finalize
    
    def get_loss(self, outputs: torch.Tensor, custom_target: torch.Tensor | None = None) -> torch.Tensor:
        ''' Note: per dim loss, not averaged '''
        if custom_target is None: 
            loss = nmse(outputs, target=self.target, target_variance=self.target_variance)
        else:
            custom_target_variance = torch.var(custom_target, dim=-1, unbiased=False) # (k,)
            loss = nmse(outputs, target=custom_target.unsqueeze(0), target_variance=custom_target_variance.unsqueeze(-1)) # (any shape, k, dims)
        return loss
    
    def get_iter_metrics(self):
        iter_fitness = self.best_term_fitness.item()
        iter_term_size = self.best_term_size
        iter_term_depth = self.best_term_depth
        return {
            'iter_fitness': [iter_fitness],
            'iter_term_size': [iter_term_size],
            'iter_term_depth': [iter_term_depth]
        }