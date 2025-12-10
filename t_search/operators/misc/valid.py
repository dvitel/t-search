from t_search.evaluators.evaluator import Evaluator
from t_search.operators.operator import Operator
from t_search.syntax import Term
from t_search.syntax.syntax import Syntax
        
class Valid(Operator):
    ''' Filters population to keep only semantically and syntactically valid terms
    '''
    def __init__(self, *,
                 syntax: Syntax,
                 evaluator: Evaluator):
        self.syntax = syntax
        self.evaluator = evaluator

    def __call__(self, population: list[Term]):
        children = [ch for ch in population if self.syntax.is_valid(ch) and self.evaluator.is_valid(ch)]
        if len(children) == 0:
            print("WARN: all population has nans or infs")
            return population
        return children
