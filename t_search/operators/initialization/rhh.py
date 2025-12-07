

from typing import Callable, Optional

import numpy as np

from t_search.syntax import Term
from t_search.syntax.syntax import Syntax
from t_search.syntax.term import Variable
from .base import Initialization
    
class RHH(Initialization):
    ''' Ramped Half and Half initialization operator '''

    def __init__(self, *, 
                
                 syntax: Syntax,
                 rnd: np.random.Generator,
                 add_metrics: Callable,

                 # from config params
                 min_depth = 1, 
                 max_depth = 5, 
                 grow_proba = 0.5,
                 leaf_proba: Optional[float] = 0.1,
                 freq_skew: bool = False):
        self.syntax = syntax
        self.rnd = rnd
        self.add_metrics = add_metrics
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.grow_proba = grow_proba
        self.leaf_proba = leaf_proba
        self.freq_skew = freq_skew

    def _rhh(self) -> Term:
        depth = self.rnd.integers(self.min_depth, self.max_depth + 1)
        leaf_prob = self.leaf_proba if self.rnd.random() < self.grow_proba else 0
        term = self.syntax.grow(depth, 
                    grow_leaf_prob=leaf_prob,
                    freq_skew=self.freq_skew)
        return term
    
    def __call__(self, pop_size: int) -> list[Term]:
        population = []
        for _ in range(pop_size):
            term = self._rhh()
            # print(str(term))
            if term is not None:
                population.append(term)
        return population

class RHHCached(RHH):

    def __init__(self, *, 
                 
                 # from solver context
                 vars: dict[str, Variable],
                 syntax: dict[tuple[str, Term], Term], # global syntax cache

                 **kwargs):
        super().__init__(**kwargs)
        self.vars = vars 
        self.syntax = syntax 

    ''' Considers inner terms of solver syntax cache '''
    def __call__(self, pop_size: int) -> list[Term]:
        none_count = 0
        sz = pop_size - len(self.vars)
        while len(self.syntax) < sz:
            term = self._rhh() # internally adds to syntax
            if term is None:
                none_count += 1
            if none_count == pop_size:
                break 
        population = list(self.syntax.values())[:sz]
        population.extend(self.vars.values())
        return population