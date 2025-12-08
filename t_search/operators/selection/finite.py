

from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.term_spatial import InvalidTerms
from .base import Operator
from t_search.syntax import Term
        
class Finite(Operator):
    ''' Selects only children that have finite or unknown outputs 
        Resorts back to full population if all outputs are infinie or nan
    '''
    def __init__(self, *,
                 invalid_terms: InvalidTerms):
        self.invalid_terms = invalid_terms

    def __call__(self, population: list[Term]):
        children = [ch for ch in population if not self.invalid_terms.is_invalid(ch)]
        if len(children) == 0:
            print("WARN: all population has nans or infs")
            return population
        return children
