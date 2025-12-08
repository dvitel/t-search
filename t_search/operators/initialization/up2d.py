from t_search.operators.initialization.base import Initialization
from t_search.syntax import Term
from t_search.syntax.syntax import Syntax

class Up2D(Initialization):
    ''' All trees (without constants) up to specified depth '''

    def __init__(self, *, 
                 
                 syntax: Syntax,
                 # from config params
                 depth = 2, 
                 max_consts: int = 0,
                ):
        self.syntax = syntax
        self.depth = depth
        self.max_consts = max_consts

    def __call__(self) -> list[Term]:
        population = self.syntax.get_all_terms(up2depth=self.depth, max_consts=self.max_consts)
        return population