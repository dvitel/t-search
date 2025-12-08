

from typing import Callable
from t_search.evaluators.evaluator import Evaluator
from t_search.operators.base import Operator
from t_search.syntax.term import Term


class SurvivorSelection(Operator):
    ''' Base class for survivor selection operators '''

    def __init__(self, *,
                 get_cur_population: Callable[[], list[Term]],
                 evaluator: Evaluator):
        self.get_cur_population = get_cur_population
        self.evaluator = evaluator