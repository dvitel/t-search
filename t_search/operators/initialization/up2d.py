

import numpy as np
from t_search.syntax import Term
from t_search.syntax.syntax import Syntax
from .base import Initialization

class Up2D(Initialization):
    ''' All trees (without constants) up to specified depth '''

    def __init__(self, *, 
                 
                 syntax: Syntax,
                 rnd: np.random.Generator,
                 # from config params
                 depth = 2, 
                 max_consts: int = 0,
                 force_pop_size: bool = False,
                ):
        self.syntax = syntax
        self.rnd = rnd
        self.depth = depth
        self.force_pop_size = force_pop_size
        self.max_consts = max_consts

    def __call__(self, pop_size: int) -> list[Term]:
        population = self.syntax.get_all_terms(up2depth=self.depth, max_consts=self.max_consts)
        if self.force_pop_size:
            if len(population) > pop_size:
                population = self.rnd.choice(population, size=pop_size, replace=False).tolist()
            elif len(population) < pop_size:
                pop_extend = pop_size - len(population)
                population.extend(self.rnd.choice(population, size=pop_extend, replace=True))
        return population