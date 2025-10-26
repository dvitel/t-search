from typing import TYPE_CHECKING, Generator, Optional

from syntax import Term, TermPos, Value, shuffled_position_flow

if TYPE_CHECKING:
    from t_search.solver import GPSolver
    
from ..mutation import TermMutation

class TermCrossover(TermMutation): 
    ''' Abstract base. Two parents crossover. Asymmetric implementation, child is produced from first parent '''

    def crossover_terms(self, solver: 'GPSolver', term: Term, other_term: Term) -> Term | None:
        ''' Abstract. Uses term as based and material from other_term to form a child ''' 
        pass # to be implemented in subclasses

    def select_mate(self, solver: 'GPSolver', term: Term) -> Term: 
        ''' Picks mate for given term. Default: random '''
        term = solver.rnd.choice(self.cur_parents)
        return term

    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:        
        other_term = self.select_mate(solver, term)
        child = self.crossover_terms(solver, term, other_term)
        return child

class PositionCrossover(TermCrossover):
    ''' Abstract base. Crossovers selected positions of two terms '''

    def __init__(self, name, *, max_pos_tries: int = 1e6, leaf_proba: Optional[float] = 0.1, 
                                exclude_values: bool = True, **kwargs):
        super().__init__(name, **kwargs)
        self.max_pos_tries = max_pos_tries    
        self.leaf_proba = leaf_proba
        self.exclude_values = exclude_values

    def crossover_positions(self, solver: 'GPSolver', term: Term, position: TermPos, other_term: Term, other_position: TermPos) -> Term | None:
        ''' Abstract. Exchanges terms at positions. '''
        pass # to be implemented in subclasses        

    def default_position_flow(self, solver: 'GPSolver', term: Term) -> Generator[TermPos]:
        positions = solver.get_positions(term)
        if self.exclude_values:
            positions = [pos for pos in positions if not isinstance(pos.term, Value)]
        flow = shuffled_position_flow(positions, self.leaf_proba, solver.rnd)
        return flow
    
    def select_position_pairs(self, solver: 'GPSolver', term: Term, other_term: Term) -> Generator[tuple[TermPos, TermPos]]:
        for pos1 in self.default_position_flow(solver, term):
            for pos2 in self.default_position_flow(solver, other_term):
                if pos1.term == pos2.term:
                    continue
                yield pos1, pos2

    def crossover_terms(self, solver: 'GPSolver', term: Term, other_term: Term) -> Term | None:

        positions = self.select_position_pairs(solver, term, other_term)
        
        pos_try = 0
        for position, other_position in positions:
            if pos_try >= self.max_pos_tries:
                break
            pos_try += 1
            mutated_term = self.crossover_positions(solver, term, position, other_term, other_position)
            if mutated_term is not None:       
                return mutated_term
            
        return None         
