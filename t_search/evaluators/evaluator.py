''' Base interface for syntax Evaluators'''

from collections.abc import Callable
from typing import Literal, NamedTuple, Optional, Sequence

import torch

from t_search.base import ServiceBase
from t_search.evaluators.optimizer import Optimizer
from t_search.operators.listeners.base import EvalListener

from .term_spatial import InvalidTerms, TermVectorStorage
from t_search.syntax.evaluation import evaluate
from t_search.syntax.term import Variable, Value, Term
from t_search.utils import EvSearchTermination, stack_rows

from .fitness import get_fitness_fns

class Evaluations(NamedTuple):
    ''' grouped evaluations of many terms '''
    term: list[Term]
    outputs: list[torch.Tensor] | torch.Tensor
    fitness: None | list[torch.Tensor] | torch.Tensor = None

class Evaluator:

    def is_const(self, outputs: torch.Tensor) -> Optional[torch.Tensor]:
        ''' Checks if variability of outputs signals is close to contant according to config'''
        pass 

    def eval(
        self,
        terms: Sequence[Term] | Term,
        *,
        return_outputs: Literal["list", "tensor"] = "list",
        return_fitness: Literal["none", "list", "tensor"] = "none",
    ) -> Evaluations:
        ''' Evaluates given terms. '''
        pass

    def eval_best(self, var_binding: dict[str, torch.Tensor], ops: dict[str, Callable] | None = None) -> torch.Tensor:
        ''' Evaluates the best term found so far with given variable bindings and operations '''
        pass

    def get_loss_fn(self, get_binding, set_binding):
        ''' Differentiable loss function aligned with fitness '''
        pass


class DefaultEvaluator(Evaluator, ServiceBase):
    ''' Default syntax term executor according to given operational semantics
        Lower fitness is better.
    '''

    def __init__(self, *,
                    target: torch.Tensor, 
                    ops: dict[str, Callable],
                    storage: TermVectorStorage, # should be injected 
                    invalid_terms: InvalidTerms,
                    add_metrics: Callable[..., None],
                    fitness_name: str = 'nmse',
                    fitness_atol: float = 1e-05,           
                    with_inner_evals: bool = True,      
                    max_root_evals: int = 100_000,
                    max_evals: int = 1_000_000,
                    eval_fn: Callable = evaluate,
                    listeners: list[EvalListener] = [],
                ):
        
        self.storage: TermVectorStorage = storage
        self.fitness_name = fitness_name
        self.fitness_atol = fitness_atol
        self.max_root_evals = max_root_evals
        self.root_evals: int = 0
        self.max_evals = max_evals
        self.evals: int = 0
        self.eval_fn = eval_fn
        self.add_metrics = add_metrics

        fitness_fn_builder = get_fitness_fns(fitness_name)
        self.fitness_fn_builder = fitness_fn_builder
        self.fitness_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x

        self.new_term_outputs: dict[Term, torch.Tensor] = {}
        self.invalid_terms = invalid_terms
        # self.const_term_outputs: dict[Term, torch.Tensor] = {}

        self.term_fitness: dict[Term, torch.Tensor] = {}
        self.with_inner_evals: bool = with_inner_evals

        self.best_term: Optional[Term] = None
        self.best_term_outputs: Optional[torch.Tensor] = None
        self.best_term_fitness: Optional[torch.Tensor] = None

        self.target: torch.Tensor = target
        self.ops: dict[str, Callable] = ops
        self.listeners: list[EvalListener] = listeners

        self.bad_fitness = torch.tensor(torch.inf, device=target.device, dtype=target.dtype)

        self.new_listener_terms: list[Term] = []

    def get_finalizer(self, add_metrics: Callable[..., None], best_term_callback: Callable[[Term], dict]):
        ''' Called on the end of solver search'''

        best_term_metrics = best_term_callback(self.best_term) if self.best_term is not None else {}

        add_metrics(
            evals = self.evals,
            root_evals = self.root_evals,            
            invalid_terms = len(self.invalid_terms.terms),
            best_term = self.best_term,
            best_fitness = self.best_term_fitness,
            **best_term_metrics
        )

        def finalizer():
            for output in self.new_term_outputs.values():
                del output
            for fitness in self.term_fitness.values():
                del fitness

        return finalizer

    def is_const(
        self,
        outputs: torch.Tensor) -> Optional[torch.Tensor]:
        """Check if any of outputs is const or very slow function """
        mean = outputs.mean(dim=-1)
        fitness = self.fitness_fn_builder(mean)(outputs)
        if fitness < self.fitness_atol:
            return mean
        return None 

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
        if self.invalid_terms.is_invalid(term):
            return self.invalid_terms.get_outputs(term)
        # if term in self.const_term_outputs:
        #     return self.const_term_outputs[term]
        return None    
    
    def _get_fitness(self, term: Term) -> torch.Tensor:
        if term in self.term_fitness:
            return self.term_fitness[term]
        elif self.invalid_terms.is_invalid(term):
            return self.bad_fitness
        raise ValueError(f"Term {term} has no fitness computed")    

    def _get_binding(self, root: Term, term: Term) -> Optional[torch.Tensor]:
        res_in_cache = self._get_cached_output(term)

        if res_in_cache is None:
            self.add_metrics(eval_cache_miss=1)
        else:
            self.add_metrics(eval_cache_hit=1)

        return res_in_cache
    
    def _default_set_binding(self, root: Term, term: Term):
        self.evals += 1
        if root == term:
            self.root_evals += 1
        if self.evals >= self.max_evals:
            raise EvSearchTermination("MAX_EVAL")
        if self.root_evals >= self.max_root_evals:
            raise EvSearchTermination("MAX_ROOT_EVAL")
    
    def _set_binding(self, root: Term, term: Term, value: torch.Tensor):
        self._default_set_binding(root, term)
        self.new_term_outputs[term] = value
    
    def _update_best_term(self, outputs: torch.Tensor, fitness: torch.Tensor):
        if len(outputs) == 0:
            return
        best_new_fitness, best_new_id = torch.min(fitness, dim=0)
        new_outputs = outputs[best_new_id]
        new_term = self.storage.get_term_for_semantics(new_outputs)
        assert new_term is not None, "New best term must be in storage"
        if (self.best_term is None) or \
            (best_new_fitness < self.best_term_fitness):
            # torch.isclose(best_new_fitness, self.best_term_fitness, atol=self.fitness_atol, rtol=0) or \
            self.best_term = new_term
            self.best_term_fitness = best_new_fitness
            self.best_term_outputs = new_outputs
        if self.best_term_fitness < self.fitness_atol:
            raise (EvSearchTermination("SOLVED"))
        
    def _eval_loop(self, terms: list[Term]) -> list[Term]:
        ''' Intrenal, evaluates given terms and all produced terms by listeners'''

        optim_terms = []    
        if len(terms) > 0:
            optim_terms = self._eval_group(self, terms)

        while len(self.new_listener_terms) > 0:
            new_terms = self.new_listener_terms
            self.new_listener_terms.clear()
            self._eval_group(new_terms)
            self.new_listener_terms.extend((t for t in self.new_listener_terms if self._get_cached_output(t) is None))

        return optim_terms

    def _eval_one(self, term: Term, 
                  get_binding: Callable | None = None,
                  set_binding: Callable | None = None) -> tuple[Term, torch.Tensor]:
        ''' Interrnal, can be overriden, one term evaluation without any caching '''
        output = self.eval_fn(term, self.ops, get_binding or self._get_binding, set_binding or self._set_binding)
        return (term, output)
    
    def _eval_group(self, terms: list[Term]):
        ''' Internal, eager eval ofo many terms without caching '''

        self.new_term_outputs.clear()

        outputs = []
        optim_terms = []
        for term in terms:
            new_term, output = self._eval_one(term)
            optim_terms.append(new_term)
            outputs.append(output)

        new_terms = [t for t in optim_terms if t in self.new_term_outputs]
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
                self.invalid_terms.add_invalid(invalid_term, outputs[infinite_id])
            new_semantics = semantics[valid_ids]
            valid_terms = [new_terms[i] for i in valid_ids.tolist()]
            del semantics, finite_semantics_mask, infinite_ids, valid_ids

            if len(valid_terms) > 0:
                semantics = new_semantics

                new_fitness: torch.Tensor = self.fitness_fn(semantics)
                self.storage.insert(valid_terms, semantics)
                for t, f in zip(valid_terms, new_fitness):
                    self.term_fitness[t] = f                
                
                for listener in self.listeners:
                    listener_terms = listener.on_eval(valid_terms, semantics, new_fitness)
                    if listener_terms is not None:
                        self.new_listener_terms.extend(listener_terms)                

                self._update_best_term(semantics, new_fitness)                

                del semantics

        self.new_term_outputs.clear()

        return optim_terms
    
    def get_loss_fn(self, get_binding):
        ''' Differentiable function for optimization that iss aligned with fitness (nmse by default) '''
        # TODO: probably define better loss_fn - we use just (f(x) - target)^2)
        def new_get_binding(root: Term, term: Term) -> Optional[torch.Tensor]:
            outputs = get_binding(root, term)
            if outputs is not None:
                return outputs
            return self._get_binding(root, term)
        def loss_fn(term: Term) -> torch.Tensor:
            outputs = self.eval_fn(term, self.ops, new_get_binding, self._default_set_binding)
            return torch.mean((outputs - self.target) ** 2, dim=-1)
        return loss_fn

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
            optim_terms = self._eval_loop(eval_terms)
            eval_outputs = [self._get_cached_output(term) for term in optim_terms]
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

    def eval_best(self, var_binding: dict[str, torch.Tensor], ops: dict[str, Callable] | None = None) -> torch.Tensor:
        if self.best_term is None:
            raise RuntimeError("Evaluator is not fitted yet")

        def get_binding(root: Term, term: Term) -> Optional[torch.Tensor]:
            if isinstance(term, Variable):
                return var_binding[term.var_id]
            if isinstance(term, Value):
                # return self.const_binding[term.value]
                return term.value
            return None

        def set_binding(*_):
            pass

        _, output = self._eval_one(self.best_term, ops, get_binding, set_binding)
        return output    
    

class OptimEvaluator(Evaluator):
    ''' Perform optimization before evaluation '''

    def __init__(self, *,
                 evaluator: Evaluator,
                 optimizer: Optimizer):
        super().__init__()
        self.evaluator = evaluator
        self.optimizer = optimizer

    def is_const(self, outputs: torch.Tensor) -> Optional[torch.Tensor]:
        return self.evaluator.is_const(outputs)

    def eval(
        self,
        terms: Sequence[Term] | Term,
        *,
        return_outputs: Literal["list", "tensor"] = "list",
        return_fitness: Literal["none", "list", "tensor"] = "none",
    ) -> Evaluations:
        optim_terms = []
        for term in terms:
            optim_term = self.optimizer.optimize(term)
            optim_terms.append(optim_term)
        res = self.evaluator.eval(
            optim_terms,
            return_outputs=return_outputs,
            return_fitness=return_fitness
        )
        return res

    def eval_best(self, var_binding: dict[str, torch.Tensor], ops: dict[str, Callable] | None = None) -> torch.Tensor:
        return self.evaluator.eval_best(var_binding, ops)

    def get_loss_fn(self, get_binding, set_binding):
        return self.evaluator.get_loss_fn(get_binding, set_binding)