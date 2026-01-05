from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.semantics import Semantics
from t_search.operators.mutation import TermMutation
from t_search.syntax.term import Term, TermPos

class ReduceConsts(TermMutation):
    ''' Removes constant semantics from terms '''

    def __init__(self, 
                 *,
                 evaluator: Evaluator,
                 semantics: Semantics,
                 fitness: Fitness,
                 **kwargs):
        super().__init__(**kwargs)
        self.evaluator = evaluator
        self.semantics = semantics
        self.fitness = fitness

    def mutate_term(self, term: Term) -> Term | None:
        if not self.semantics.is_valid(term):
            return None
        
        already_checked = {}
        def replace_consts(cur_term: Term, *_) -> Term:
            if cur_term in already_checked:
                return already_checked[cur_term]
            semantics = self.semantics.get_outputs(cur_term)
            if semantics is None:
                _, semantics = self.evaluator.eval(cur_term)
            is_const = self.semantics.is_const(semantics)
            if is_const is not None:
                new_term = self.syntax.get_const(value=is_const)
            else:
                new_term = None 
            already_checked[cur_term] = new_term
            return new_term
        new_term = self.syntax.replace_fn(term, replace_consts) 
        if not self.syntax.is_valid(new_term):
            return None
        if new_term == term:
            return None      
        
        self.semantics.copy_binding(term, new_term)
        self.fitness.copy_fitness(term, new_term)

        return new_term

