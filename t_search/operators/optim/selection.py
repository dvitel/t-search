from typing import Sequence

import torch

from t_search.evaluators.fitness import Fitness
from t_search.operators.classic.ts import TS
from t_search.syntax.syntax import Syntax
from t_search.syntax.term import Term, Value

class FrontierSelection(TS):
    ''' TS applied to frontier of optimization paths  '''
    def __init__(self,
                 syntax: Syntax,
                 term_frontier: set[Term],
                 with_subterms: bool = False,
                  **kwargs):
        super().__init__(**kwargs)
        self.term_frontier = term_frontier
        self.with_subterms = with_subterms
        self.syntax = syntax

    def __call__(self, population: Sequence[Term]) -> Sequence[Term]:
        all_terms = set(self.term_frontier)
        all_terms.update(population)
        if self.with_subterms:
            new_all_terms = set()
            for term in all_terms:
                term_subterms = set([term])
                for pos in self.syntax.get_positions(term):
                    if not isinstance(pos.term, Value):
                        term_subterms.add(pos.term)
                subterms = list(term_subterms)
                fitness = self.fitness.get_fitness(subterms, return_type="tensor")
                best_fitness_id = torch.argmin(fitness)
                best_term = subterms[best_fitness_id]
                new_all_terms.add(best_term)
                del fitness 
            all_terms = new_all_terms
        children = super().__call__(list(all_terms), size=self.selection_size)
        # children = list(self.term_frontier)
        # left_size = self.selection_size - len(children)
        # other_children = super().__call__(population, size=left_size) 
        # children.extend(other_children)
        return children