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
                ):
        self.syntax = syntax
        self.depth = depth
        self.max_consts = max_consts
        self.with_free_vars = with_free_vars

    def __call__(self) -> list[Term]:
        population = self.syntax.get_all_terms(up2depth=self.depth, max_consts=self.max_consts)
        if self.with_free_vars:
            vars = self.syntax.get_vars()
            population.extend(vars)
        # tt = self.syntax.get_op("add", self.syntax.get_var("x0"), self.syntax.get_const(1.0))
        # population.append(tt)
        return population