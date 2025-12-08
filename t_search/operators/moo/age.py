''' (fitness, Age)-based survivor selection '''

from t_search.operators.survivor_selection import SurvivorSelection
from t_search.syntax.term import Term


class AgeSurvivorSelection(SurvivorSelection):

    def __call__(self, offspring: list[Term]) -> list[Term]:
        pass