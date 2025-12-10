from typing import Callable
from t_search.evaluators.evaluator import Evaluator
from t_search.operators.operator import Operator
from t_search.syntax.term import Term


class SurvivorSelection(Operator):
    ''' Base class for survivor selection operators '''

    def __init__(self, *,
                 get_cur_population: Callable[[], list[Term]]):
        self.get_cur_population = get_cur_population

    def select(self, parents: list[Term], offspring: list[Term]) -> list[Term]:
        ''' Select survivors from parents and offspring '''
        raise NotImplementedError()

    def __call__(self, offspring: list[Term]) -> list[Term]:
        parents = self.get_cur_population()
        new_population = self.select(parents, offspring)
        return new_population