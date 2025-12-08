from t_search.operators.crossover import PositionCrossover
from t_search.syntax import Term, TermPos

class RPX(PositionCrossover):
    ''' One Random Position Crossover '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.crossover_cache: dict[tuple[Term, Term, int, Term], Term] = {}    

    def crossover_positions(self, term: Term, position: TermPos, 
                                    other_term: Term, other_position: TermPos) -> Term | None:

        crossover_key = (term, position.term, position.occur, other_position.term)
        if crossover_key in self.crossover_cache:
            child = self.crossover_cache[crossover_key]
            return child 

        child = self.syntax.replace_position(term, position, other_position.term)
        
        return child
