from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.semantics import Semantics
from t_search.operators.mutation import TermMutation
from t_search.syntax import Term, TermPos
from t_search.syntax.term import Value

class BestSubterm(TermMutation):
    """Replaces parent with best subterm"""

    def __init__(self, *, 
                 evaluator: Evaluator,
                 semantics: Semantics,
                 fitness: Fitness,
                 test_based: bool = False,
                 test_based_percent: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.evaluator = evaluator
        self.semantics = semantics
        self.fitness = fitness
        self.test_based = test_based
        self.test_based_percent = test_based_percent

    def mutate_term(
        self, term: Term
    ) -> Term | None:
        
        self.evaluator.eval(term)
        subterms = [pos.term for pos in self.syntax.get_positions(term) if not isinstance(pos.term, Value)] # only non-leaf subterms are considered
        subterms.append(term)
        
        if not self.test_based: # fitness is used 

            fitness = self.fitness.get_fitness(subterms, return_type="tensor")
            best_term_id = fitness.argmin().item()
            mutated_term = subterms[best_term_id]
            del fitness
            return mutated_term
        
        else: 

            outputs = self.semantics.get_semantics(subterms)
            loss = self.fitness.get_loss(outputs) # per test 
            subterm_mask = loss < loss[-1].unsqueeze(0)
            subterm_count = subterm_mask.sum(dim=-1) # number of better tests 
            best_term_id = subterm_count.argmax().item()
            best_count = subterm_count[best_term_id].item()
            if best_count > self.test_based_percent * loss.shape[-1]: 
                mutated_term = subterms[best_term_id]
                return mutated_term

        return term
