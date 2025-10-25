''' Base interface for mutation operators. '''

from typing import TYPE_CHECKING, Generator, Optional, Sequence

from ..base import Operator
from t_search.term import Term, TermPos, shuffled_position_flow

if TYPE_CHECKING:
    from t_search.solver import GPSolver  # Import only for type checking

class TermMutation(Operator): 
    ''' Abstract base. Mutates population one term at a time (1-to-1 mapping pattern to repr or mutated child)'''
    def __init__(self, name, *, rate : float = 0.1, **kwargs):
        super().__init__(name, **kwargs)
        self.rate = rate
        self.cur_parents = None

    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:
        ''' Abstract. Mutates one term in the context of parents and already generated children ''' 
        pass # to be implemented in subclasses

    def exec(self, solver: 'GPSolver', population: Sequence[Term]) -> Sequence[Term]: 
        ''' 
            Some mutations could return None, we would like to reattempt if small number was mutated t guarantee mutated_size.
            However, we still stick to only one pass through population.
        '''

        self.cur_parents = population

        success = 0
        fail = 0     
        repr_cnt = 0    

        size = len(population)
        mutated_size = int(self.rate * size)
        permuted_term_ids = solver.rnd.permutation(size)         
        children = [] 

        for term_id in permuted_term_ids:
            term = population[term_id]
            if mutated_size <= 0: 
                children.append(term)
                repr_cnt += 1
            else: 
                child = self.mutate_term(solver, term)
                if child is not None:
                    success += 1
                    children.append(child)
                    mutated_size -= 1
                else:
                    fail += 1
                    children.append(term)

        self.metrics["success"] = self.metrics.get("success", 0) + success
        self.metrics["fail"] = self.metrics.get("fail", 0) + fail
        self.metrics["repr"] = self.metrics.get("repr", 0) + repr_cnt
        
        return children
    
class PositionMutation(TermMutation):
    ''' Abstract base. Mutates specific position inside a term. '''

    def __init__(self, name, *, max_pos_tries: int = 1e6, leaf_proba: Optional[float] = 0.1, **kwargs):
        super().__init__(name, **kwargs)
        self.max_pos_tries = max_pos_tries
        self.leaf_proba = leaf_proba

    def select_positions(self, solver: 'GPSolver', term: Term) -> Generator[TermPos]:   
        positions = solver.get_positions(term)
        return shuffled_position_flow(positions, self.leaf_proba, solver.rnd)

    def mutate_position(self, solver: 'GPSolver', term: Term, position: TermPos) -> Term | None:
        ''' Abstract. Mutates term at the given position. '''
        pass # to be implemented in subclasses    

    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:
        ''' Mutates one term in the context of parents and already generated children ''' 
        
        positions = self.select_positions(solver, term)
        
        pos_try = 0
        for position in positions:
            if pos_try >= self.max_pos_tries:
                break
            pos_try += 1
            mutated_term = self.mutate_position(solver, term, position)
            if mutated_term is not None:       
                return mutated_term
            
        return None 
