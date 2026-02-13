

from typing import Callable, Optional

import numpy as np

from t_search.operators.initialization import Initialization
from t_search.syntax import Term
from t_search.syntax.syntax import Syntax
from t_search.syntax.term import Variable
    
class RHH(Initialization):
    ''' Ramped Half and Half initialization operator '''

    def __init__(self, *, 
                
                 syntax: Syntax,
                 rnd: np.random.Generator,
                 add_metrics: Callable,

                 # from config params
                 size: int = 1000,
                 min_depth = 1, 
                 max_depth = 5, 
                 grow_proba = 0.5,
                 syntactic_duplicate_retries: int = 1,
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
        self.size = size
        self.syntactic_duplicate_retries = syntactic_duplicate_retries

    def _rhh(self) -> Term:
        depth = self.rnd.integers(self.min_depth, self.max_depth + 1)
        leaf_prob = self.leaf_proba if self.rnd.random() < self.grow_proba else 0
        term = self.syntax.grow(depth, 
                    grow_leaf_prob=leaf_prob,
                    freq_skew=self.freq_skew)
        return term
    
    def __call__(self) -> list[Term]:
        population = []
        present_terms = set()
        for _ in range(self.size):
            last_good_term = None
            for _ in range(self.syntactic_duplicate_retries):
                term = self._rhh()
                # print(str(term))
                if term is not None:
                    last_good_term = term
                    if term in present_terms:
                        continue
                    present_terms.add(term)
                    population.append(term)
                    last_good_term = None
                    break
            if last_good_term is not None:
                population.append(last_good_term)
                present_terms.add(last_good_term)
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
    def __call__(self) -> list[Term]:
        none_count = 0
        sz = self.size - len(self.vars)
        while len(self.syntax) < sz:
            term = self._rhh() # internally adds to syntax
            if term is None:
                none_count += 1
            if none_count == self.size:
                break 
        population = list(self.syntax.values())[:sz]
        population.extend(self.vars.values())
        return population