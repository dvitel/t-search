''' Base interface for syntax Evaluators'''

from collections.abc import Callable
from typing import Literal, NamedTuple, Optional, TYPE_CHECKING, Sequence

import torch

from t_search.evaluators.term_spatial import TermVectorStorage
from t_search.operators.listeners.base import Listener
from t_search.spatial import VectorStorage
from t_search.syntax.evaluation import evaluate
from t_search.syntax.term import Variable, Value, Term
from t_search.utils import EvSearchTermination, stack_rows

from .fitness import get_fitness_fns

if TYPE_CHECKING:
    from t_search.solver import GPSolver

class Evaluation(NamedTuple):
    ''' evaluation of one term '''
    sketch: Term # skeleton from which the concrete term was created, by default it is term itself
    term: Term # same or revised term under evaluation
    outputs: torch.Tensor # per test outcomes 
    fitness: torch.Tensor # aggregated outcomes

class Evaluations(NamedTuple):
    ''' grouped evaluations of many terms '''
    outputs: list[torch.Tensor] | torch.Tensor
    fitness: None | list[torch.Tensor] | torch.Tensor = None

class Evaluator:
    ''' Default syntax term executor according to given operational semantics
        Lower fitness is better.
    '''

    def __init__(self, *,
                    vector_storage: VectorStorage, # should be injected 
                    fitness_name: str = 'nmse',
                    fitness_atol: float = 1e-05,           
                    with_inner_evals: bool = True,      
                    max_root_evals: int = 100_000,
                    max_evals: int = 1_000_000,
                    # output_rtol=1e-04,
                    # output_atol=1e-04, 
                ):

        self.storage: TermVectorStorage = TermVectorStorage(vector_storage)
        self.fitness_name = fitness_name
        self.fitness_atol = fitness_atol
        self.max_root_evals = max_root_evals
        self.root_evals: int = 0
        self.max_evals = max_evals
        self.evals: int = 0

        fitness_fn_builder = get_fitness_fns(fitness_name)
        self.fitness_fn_builder = fitness_fn_builder
        self.fitness_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x

        self.new_term_outputs: dict[Term, torch.Tensor] = {}
        self.invalid_term_outputs: dict[Term, torch.Tensor] = {}
        # self.const_term_outputs: dict[Term, torch.Tensor] = {}

        self.term_fitness: dict[Term, torch.Tensor] = {}
        self.with_inner_evals: bool = with_inner_evals

        self.best_eval: Optional[Evaluation] = None

        # settings set by GPSolver in on_start 
        self.target: Optional[torch.Tensor] = None
        self.ops: Optional[dict[str, Callable]] = None
        self.listeners: list[Listener] = []

    def _clean_eval_caches(self):
        self.evals = 0
        self.root_evals = 0
        self.storage.reset()
        for output in self.new_term_outputs.values():
            del output
        self.new_term_outputs.clear()
        for output in self.invalid_term_outputs.values():
            del output
        self.invalid_term_outputs.clear()
        for fitness in self.term_fitness.values():
            del fitness
        self.term_fitness.clear()        
        self.new_listener_terms: list[Term] = []
        self.eq_group_term_order: Callable[[Term], float] = lambda t: 0.0

    def on_start(self, solver: 'GPSolver'):
        ''' Called before solver starts fitting '''
        self._clean_eval_caches()
        self.ops = solver.ops
        self.target = solver.target
        self.fitness_fn = self.fitness_fn_builder(self.target)
        self.listeners = solver.listeners
        self.new_listener_terms.clear()
        self.eq_group_term_order = lambda t: solver.get_size(t)

    def on_end(self, solver: 'GPSolver'):
        ''' Called on the end of solver search'''

        best_depth = solver.get_depth(self.best_eval.term)
        best_size = solver.get_size(self.best_eval.term)
        best_counts = solver.get_counts(self.best_eval.term)

        solver.add_metric(
            evals = self.evals,
            root_evals = self.root_evals,            
            invalid_terms = len(self.invalid_term_outputs),
            best_sketch = self.best_eval.sketch,
            best_term = self.best_eval.term,
            best_fitness = self.best_eval.fitness,
            best_term_depth = best_depth,
            best_term_size = best_size,
            best_term_counts = best_counts.tolist()
        )

        self._clean_eval_caches()
        self.new_listener_terms.clear()

    def is_const(
        self,
        outputs: torch.Tensor) -> Optional[torch.Tensor]:
        """Check if any of outputs is const or very slow function """
        mean = outputs.mean(dim=-1)
        fitness = self.fitness_fn_builder(mean)(outputs)
        if fitness < self.fitness_atol:
            return mean
        return None 
    
    def is_invalid(
        self,
        term: Term 
    ):
        return term in self.invalid_term_outputs
    
    def detect_const_range(self, var_bindings: Sequence[torch.Tensor]) -> torch.Tensor:
        min_value = self.target.min()
        max_value = self.target.max()
        if torch.isclose(
            min_value,
            max_value,
            atol=self.fitness_atol,
        ):
            min_value = min_value - 0.1
            max_value = max_value + 0.1
        const_range = torch.tensor([min_value, max_value], dtype=self.target.dtype, device=self.target.device)
        free_vars_as_one = torch.stack(tuple(var_bindings), dim=0)
        min_fv = torch.min(free_vars_as_one)
        max_fv = torch.max(free_vars_as_one)
        const_range[0] = torch.minimum(const_range[0], min_fv)
        const_range[1] = torch.maximum(const_range[1], max_fv)
        dist = const_range[0] - const_range[1]
        const_range[0] -= 0.1 * dist
        const_range[1] += 0.1 * dist
        return const_range

    def _get_cached_output(self, term: Term) -> Optional[torch.Tensor]:
        if isinstance(term, Variable):
            return self.var_binding[term.var_id]
        if isinstance(term, Value):
            # return self.const_binding[term.value]
            return term.value
        term_semantics = self.storage.get_semantics_for_term(term)
        if term_semantics is not None:
            return term_semantics
        if term in self.new_term_outputs:
            return self.new_term_outputs[term]
        if term in self.invalid_term_outputs:
            return self.invalid_term_outputs[term]
        # if term in self.const_term_outputs:
        #     return self.const_term_outputs[term]
        return None    
    
    def _get_fitness(self, term: Term) -> torch.Tensor:
        if term in self.term_fitness:
            return self.term_fitness[term]
        elif term in self.invalid_term_outputs:
            return self.bad_fitness
        raise ValueError(f"Term {term} has no fitness computed")    

    def _get_binding(self, root: Term, term: Term) -> Optional[torch.Tensor]:
        res_in_cache = self._get_cached_output(term)

        if res_in_cache is None:
            self.metrics["eval_cache_miss"] = self.metrics.get("eval_cache_miss", 0) + 1
        else:
            self.metrics["eval_cache_hit"] = self.metrics.get("eval_cache_hit", 0) + 1

        return res_in_cache
    
    def _set_binding(self, root: Term, term: Term, value: torch.Tensor):
        self.evals += 1
        if root == term:
            self.root_evals += 1
        if self.evals >= self.max_evals:
            raise EvSearchTermination("MAX_EVAL")
        if self.root_evals >= self.max_root_evals:
            raise EvSearchTermination("MAX_ROOT_EVAL")
        self.new_term_outputs[term] = value
    
    def _update_best_term(self, solver: 'GPSolver', new_terms: Sequence[Term], outputs: torch.Tensor, fitness: torch.Tensor):
        if len(new_terms) == 0:
            return
        best_new_fitness, best_new_id = torch.min(fitness, dim=0)
        new_term = new_terms[best_new_id]
        new_outputs = outputs[best_new_id]
        if self.best_eval is None:
            self.best_eval = Evaluation(new_term, new_term, best_new_fitness, new_outputs)
        elif torch.isclose(best_new_fitness, self.best_eval.fitness, atol=self.fitness_atol, rtol=0):
            best_term_size = solver.get_size(self.best_eval.term)
            new_term_size = solver.get_size(new_term)
            if new_term_size < best_term_size:
                self.best_eval = Evaluation(new_term, new_term, best_new_fitness, new_outputs)
        elif best_new_fitness < self.best_eval.fitness:
            self.best_eval = Evaluation(new_term, new_term, best_new_fitness, new_outputs)
        if self.best_eval.fitness < self.fitness_atol:
            raise (EvSearchTermination("SOLVED"))
        
    def _eval_loop(self, terms: list[Term]):
        ''' Intrenal, evaluates given terms and all produced terms by listeners'''

        if len(terms) > 0:
            self._eval_group(self, terms)

        while len(self.new_listener_terms) > 0:
            new_terms = self.new_listener_terms
            self.new_listener_terms.clear()
            self._eval_group(new_terms)
            self.new_listener_terms.extend((t for t in self.new_listener_terms if self._get_cached_output(t) is None))
        pass     

    def _eval_one(self, term: Term, 
                  ops: dict[str, Callable] | None = None,
                  get_binding: Callable | None = None,
                  set_binding: Callable | None = None) -> Optional[torch.Tensor]:
        ''' Interrnal, can be overriden, one term evaluation without any caching '''
        output = evaluate(term, ops or self.ops, get_binding or self._get_binding, set_binding or self._set_binding)
        return output
    
    def _eval_group(self, terms: list[Term]):
        ''' Internal, eager eval ofo many terms without caching '''

        self.new_term_outputs.clear()

        outputs = []
        for term in terms:
            output = self._eval_one(term)
            outputs.append(output)

        new_terms = [t for t in terms if t in self.new_term_outputs]
        if self.with_inner_evals:
            new_terms = list(self.new_term_outputs.keys())

        outputs = [self.new_term_outputs[t] for t in new_terms]
        if len(outputs) > 0:
            semantics = stack_rows(outputs, self.target)
            finite_semantics_mask = torch.isfinite(semantics).all(dim=-1)  # we do not insert nans and infs
            (valid_ids,) = torch.where(finite_semantics_mask)
            (infinite_ids,) = torch.where(~finite_semantics_mask)
            for infinite_id in infinite_ids.tolist():
                invalid_term = new_terms[infinite_id]
                self.invalid_term_outputs[invalid_term] = outputs[infinite_id]
            new_semantics = semantics[valid_ids]
            valid_terms = [new_terms[i] for i in valid_ids.tolist()]
            del semantics, finite_semantics_mask, infinite_ids, valid_ids

            if len(valid_terms) > 0:
                semantics = new_semantics

                new_fitness: torch.Tensor = self.fitness_fn(semantics)
                self.storage.insert(valid_terms, semantics, self.eq_group_term_order)
                for t, f in zip(valid_terms, new_fitness):
                    self.term_fitness[t] = f                
                
                # solver.on_eval(valid_terms, semantics, new_fitness)
                for listener in self.listeners:
                    listener_terms = listener.on_eval(self, valid_terms, semantics, new_fitness)
                    if listener_terms is not None:
                        self.new_listener_terms.extend(listener_terms)                

                self._update_best_term(valid_terms, new_fitness)                

                del semantics

        self.new_term_outputs.clear()

        return

    def eval(
        self,
        terms: Sequence[Term] | Term,
        *,
        return_outputs: Literal["list", "tensor"] = "list",
        return_fitness: Literal["none", "list", "tensor"] = "none",
    ) -> Evaluations:
        """Evaluates given terms. If terms are already in cache, results returned without affecting the metrics.
        Calls _eval internally, therefore could cause an avalanche of evaluations of new terms through listeners.
        """
        if isinstance(terms, Term):
            terms = [terms]
        outputs = [self._get_cached_output(term) for term in terms]
        eval_ids = [i for i, output in enumerate(outputs) if output is None]
        eval_terms = [terms[i] for i in eval_ids]
        if len(eval_terms) > 0:
            self._eval_loop(eval_terms)
            eval_outputs = [self._get_cached_output(term) for term in eval_terms]
            for i, eval_output in zip(eval_ids, eval_outputs):
                outputs[i] = eval_output
        output_res: list | torch.Tensor = outputs
        if return_outputs == "tensor":
            output_res = stack_rows(outputs, self.target)
        fitness_res: None | list | torch.Tensor = None
        if return_fitness != "none":
            fitness = [self._get_fitness(term) for term in terms]
            fitness_res = fitness
            if return_fitness == "tensor":
                fitness_res = torch.stack(fitness, dim=0)
        return Evaluations(output_res, fitness_res)
    
    def report_evals(self, num_evals: int, num_root_evals: int):
        self.evals += num_evals
        self.root_evals += num_root_evals
        if self.evals > self.max_evals:
            raise EvSearchTermination("MAX_EVAL", "Maximum number of evaluations reached")
        if self.root_evals > self.max_root_evals:
            raise EvSearchTermination("MAX_ROOT_EVAL", "Maximum number of root evaluations reached")    
        

    def predict(self, var_binding: dict[str, torch.Tensor], ops: dict[str, Callable] | None = None) -> torch.Tensor:
        if self.best_eval is None:
            raise RuntimeError("Solver is not fitted yet")

        def get_binding(root: Term, term: Term) -> Optional[torch.Tensor]:
            if isinstance(term, Variable):
                return var_binding[term.var_id]
            if isinstance(term, Value):
                # return self.const_binding[term.value]
                return term.value
            return None

        def set_binding(*_):
            pass

        output = self._eval_one(self.best_eval.term, ops, get_binding, set_binding)
        return output    