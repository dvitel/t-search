from typing import Generator, Optional

from t_search.operators.mutation import TermMutation
from t_search.syntax import Term, TermPos, Value
from t_search.syntax.flow import shuffled_position_flow

class TermCrossover(TermMutation): 
    ''' Abstract base. Two parents crossover. Asymmetric implementation, child is produced from first parent '''

    def crossover_terms(self, term: Term, other_term: Term) -> Term | None:
        ''' Abstract. Uses term as based and material from other_term to form a child ''' 
        pass # to be implemented in subclasses

    def select_mate(self, term: Term) -> Term | None: 
        ''' Picks mate for given term. Default: random '''
        term = self.rnd.choice(self.cur_parents)
        return term

    def mutate_term(self, term: Term) -> Term | None:        
        other_term = self.select_mate(term)
        if other_term is None:
            return None
        child = self.crossover_terms(term, other_term)
        return child

class PositionCrossover(TermCrossover):
    ''' Abstract base. Crossovers selected positions of two terms '''

    def __init__(self, *, max_pos_tries: int = 1e6, leaf_proba: Optional[float] = 0.1, 
                                exclude_values: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.max_pos_tries = max_pos_tries    
        self.leaf_proba = leaf_proba
        self.exclude_values = exclude_values

    def crossover_positions(self, term: Term, position: TermPos, other_term: Term, other_position: TermPos) -> Term | None:
        ''' Abstract. Exchanges terms at positions. '''
        pass # to be implemented in subclasses        

    def default_position_flow(self, term: Term) -> Generator[TermPos, None, None]:
        positions = self.syntax.get_positions(term)
        if self.exclude_values:
            positions = [pos for pos in positions if not isinstance(pos.term, Value)]
        flow = shuffled_position_flow(positions, self.leaf_proba, self.rnd)
        return flow
    
    def select_position_pairs(self, term: Term, other_term: Term) -> Generator[tuple[TermPos, TermPos], None, None]:
        for pos1 in self.default_position_flow(term):
            for pos2 in self.default_position_flow(other_term):
                if pos1.term == pos2.term:
                    continue
                yield pos1, pos2

    def crossover_terms(self, term: Term, other_term: Term) -> Term | None:

        positions = self.select_position_pairs(term, other_term)
        
        pos_try = 0
        for position, other_position in positions:
            if pos_try >= self.max_pos_tries:
                break
            pos_try += 1
            mutated_term = self.crossover_positions(term, position, other_term, other_position)
            if mutated_term is not None:       
                return mutated_term
            
        return None         
