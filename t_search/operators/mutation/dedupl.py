
from ..base import Operator
from t_search.syntax import Term

class Dedupl(Operator):
    ''' Removes duplicate syntaxes from the population '''

    def __call__(self, population: list[Term]) -> list[Term]:
        present_terms = set()
        children = []
        for term in population:
            if term not in present_terms:
                children.append(term)
                present_terms.add(term)
        return children