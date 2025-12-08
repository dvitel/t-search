''' Base interfaces for different evolutionary operators '''

from t_search.syntax import Term

class Initialization:

    def __call__(self) -> list[Term]:
        """Use to trigger initialization """
        pass    
    
        