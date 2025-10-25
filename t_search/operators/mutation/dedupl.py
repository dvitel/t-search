
from typing import Sequence
from ..base import Operator
from t_search.term import Term

class Dedupl(Operator):
    ''' Removes duplicate syntaxes from the population '''

    def __init__(self, name: str = "dedupl", **kwargs):
        super().__init__(name, **kwargs)    

    def exec(self, _, population: Sequence[Term]) -> Sequence[Term]:
        present_terms = set()
        children = []
        for term in population:
            if term not in present_terms:
                children.append(term)
                present_terms.add(term)
        return children