''' Selection of survivors based on Pareto front layering (NSGA-II, NSGA-III) '''

from t_search.operators.survivor_selection import SurvivorSelection
from t_search.syntax.term import Term


class NsgaSurvivorSelection(SurvivorSelection):
    ''' NSGA Survivor Selection '''

    def __call__(self, offspring: list[Term]) -> list[Term]:
        pass