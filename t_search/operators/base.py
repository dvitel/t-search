''' Base interfaces for different evolutionary operators '''

from t_search.syntax import Term

class Operator:
    ''' Base class for evolutionary operators '''
    def __call__(self, population: list[Term]) -> list[Term]:
        pass