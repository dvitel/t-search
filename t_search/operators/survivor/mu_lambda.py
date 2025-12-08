''' μ+λ and (μ,λ) survivor selection operators '''

from t_search.operators.survivor.base import SurvivorSelection
from t_search.syntax.term import Term
from t_search.utils import sorted_by_fitness

class MuLambdaSurvivorSelection(SurvivorSelection):
    ''' μ+λ, (μ,λ) Survivor Selection

    This operator selects the next generation of individuals by combining the current population (μ)
    with the newly generated offspring (λ) and selecting the best individuals based on their fitness.

    Args:
        mu (int): Number of individuals to select for the next generation.
        strict (bool): If True, raises an error if there are not enough individuals to select.
    '''

    def __init__(self, *,
                 mu: int,
                 strict: bool = True,
                 elitism: int = 0, # absolute elitism from parents
                 combine: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.strict = strict
        self.combine = combine
        self.elitism = elitism

    def __call__(self, offspring: list[Term]) -> list[Term]:
        parents = self.get_cur_population()
        if self.combine:
            combined = parents + offspring
        elif self.elitism > 0:
            fitness = self.evaluator.eval(parents, return_fitness="tensor").fitness
            elite_parents = sorted_by_fitness(parents, fitness, max_num=self.elitism)
            del fitness 
            combined = elite_parents + offspring
        else:
            combined = offspring

        if len(combined) <= self.mu:
            if self.strict and len(combined) < self.mu:
                raise ValueError(f"Not enough individuals to select {self.mu} individuals.")
            return combined
        fitness = self.evaluator.eval(combined, return_fitness="tensor").fitness
        new_population = sorted_by_fitness(combined, fitness, max_num=self.mu)
        del fitness
        return new_population