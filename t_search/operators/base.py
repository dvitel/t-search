''' Base interfaces for different evolutionary operators '''

from t_search.syntax import Term

class Operator:
    
    def exec(self, population: list[Term]) -> list[Term]:
        ''' Executes only this operator and update existing metrics state '''
        return population
    
    def call_next(self, population: list[Term], next_ops: list['Operator'] = []) -> list[Term]:
        if len(next_ops) > 0:
            next_op, *rest_ops = next_ops
            children = next_op(children, rest_ops)        
            return children
        return population

    def __call__(self, population: list[Term], next_ops: list['Operator'] = []) -> list[Term]:
        ''' Executes operator in the chain '''
        children = self.exec(population)
        children = self.call_next(children, next_ops)
        return children