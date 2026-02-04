''' Reduction of the term based on registered semantics '''

from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.semantics import Semantics
from t_search.operators.mutation import TermMutation
from t_search.syntax.term import Term


class SReduce(TermMutation):
    ''' Semantic Reduction Operator 
        Replaces subterms with semantically equivalent but syntactically simpler terms
    '''
    def __init__(self, *, 
                    semantics: Semantics,
                    evaluator: Evaluator,
                    **kwargs):
        super().__init__(**kwargs)
        self.semantics = semantics
        self.evaluator = evaluator

    def mutate_term(self, term) -> Term | None:
        def replace_with_repr(term: Term, *_):
            repr_term = self.semantics.get_repr_for_term(term)
            if repr_term is not None and repr_term != term:
                return repr_term
        simplified = self.syntax.replace_fn(term, replace_with_repr)
        if self.debug and simplified != term:
            print(f"SRed:\n\t{term} --->\n\t{simplified}")
        return simplified
    
    def __call__(self, population):
        if self.debug:
            print("SReduce start")
        new_pop = super().__call__(population)
        to_reeval = [ t for t in new_pop if self.semantics.get_binding(t) is None ]
        if len(to_reeval) > 0:
            self.evaluator.eval(to_reeval)
        # filtered = [t for t in new_pop if self.syntax.is_valid(t)]
        # return filtered
        if self.debug:
            print("SReduce end")        
        return new_pop