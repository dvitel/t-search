''' Base interfaces for different evolutionary operators '''

from t_search.syntax import Term

class Operator:
    ''' Base class for evolutionary operators '''
    def __call__(self, population: list[Term]) -> list[Term]:
        pass

class Initialization:

    def __call__(self) -> list[Term]:
        """Use to trigger initialization """
        pass    
    
        