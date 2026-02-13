from t_search.operators.initialization import Initialization
from t_search.syntax import Term
from t_search.syntax.syntax import Syntax

class Up2D(Initialization):
    ''' All trees (without constants) up to specified depth '''

    def __init__(self, *, 
                 
                 syntax: Syntax,
                 # from config params
                 with_free_vars: bool = False,
                 depth = 2, 
                 max_consts: int = 0,
                 const_1: bool = False
                ):
        self.syntax = syntax
        self.depth = depth
        self.max_consts = max_consts
        self.with_free_vars = with_free_vars
        self.const_1 = const_1

    def __call__(self) -> list[Term]:
        population = self.syntax.get_all_terms(up2depth=self.depth, max_consts=self.max_consts, const_1=self.const_1)
        if self.with_free_vars:
            vars = self.syntax.get_vars()
            population.extend(vars)
        return population