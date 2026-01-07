''' Base interface for syntax Evaluators'''

from collections.abc import Callable
from typing import Literal, Optional, Sequence

import torch

from t_search.base import ServiceBase
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.optimizer import Optimizer

from t_search.evaluators.semantics import Semantics
from t_search.operators.listeners import EvalListener
from t_search.operators.operator import Operator
from t_search.syntax.evaluation import evaluate
from t_search.syntax.term import Term
from t_search.utils import EvSearchTermination, stack_rows


# class Evaluations(NamedTuple):
#     ''' grouped evaluations of many terms '''
#     term: list[Term]
#     outputs: list[torch.Tensor] | torch.Tensor
#     fitness: None | list[torch.Tensor] | torch.Tensor = None

class Evaluator(Operator):

    # def is_const(self, outputs: torch.Tensor) -> Optional[torch.Tensor]:
    #     ''' Checks if variability of outputs signals is close to contant according to config'''
    #     pass 

    # def is_valid(self, term: Term) -> bool:
    #     ''' Checks validity of term according to semantic constraints '''
    #     pass

    def test(self, get_binding: Callable) -> torch.Tensor:    
        ''' Test mode evaluation with test bindings '''
        pass

    def eval(
        self,
        terms: Sequence[Term] | Term,
        # *,
        # return_outputs: Literal["list", "tensor"] = "list",
        # return_fitness: Literal["none", "list", "tensor"] = "none",
    ) -> list[tuple[Term, torch.Tensor]] | tuple[Term, torch.Tensor]:
        ''' Evaluates given terms. '''
        pass

    # def eval_best(self, var_bindings: dict[str, torch.Tensor], ops: dict[str, Callable] | None = None) -> torch.Tensor:
    #     ''' Evaluates the best term found so far with given variable bindings and operations '''
    #     pass

    def get_loss_fn(self, get_binding: Callable | None = None, set_binding: Callable | None = None, no_cache: bool = False):
        ''' Differentiable loss function aligned with fitness '''
        pass

    def __call__(self, population: Sequence[Term]) -> Sequence[Term]:
        ''' Evaluate given population of terms and return them'''
        res = self.eval(population)
        children = [t for t, _ in res]
        return children


class DefaultEvaluator(Evaluator, ServiceBase):
    ''' Default syntax term executor according to given operational semantics
        Lower fitness is better.
    '''

    def __init__(self, *,
                    # target: torch.Tensor, 
                    # var_bindings: dict[str, torch.Tensor],
                    semantics: Semantics,
                    fitness: Fitness,
                    dims: int,
                    ops: dict[str, Callable],
                    # storage: TermVectorStorage, # should be injected 
                    add_metrics: Callable[..., None],
                    # fitness_name: str = 'nmse',
                    # fitness_atol: float = 1e-05,           
                    with_inner_evals: bool = True,      
                    max_root_evals: int = 100_000,
                    max_evals: int = 1_000_000,
                    # eval_fn: Callable = evaluate,
                    listeners: list[EvalListener] = [],
                    device: torch.device
                ):
        
        # self.storage: TermVectorStorage = storage
        # self.fitness_name = fitness_name
        # self.fitness_atol = fitness_atol
        self.semantics = semantics
        self.fitness = fitness
        self.dims = dims
        self.max_root_evals = max_root_evals
        self.root_evals: int = 0
        self.max_evals = max_evals
        self.evals: int = 0
        self.eval_calls: int = 0
        # self.eval_fn = eval_fn
        self.add_metrics = add_metrics
        self.is_cuda = device.type == 'cuda'
        self.eval_cache_hits: int = 0
        self.eval_cache_miss: int = 0

        # fitness_fn_builder = get_fitness_fns(fitness_name)
        # self.fitness_fn_builder = fitness_fn_builder
        # self.fitness_fn: Callable[[torch.Tensor], torch.Tensor] = fitness_fn_builder(target)

        self.new_term_outputs: dict[Term, torch.Tensor] = {}        
        # self.const_term_outputs: dict[Term, torch.Tensor] = {}

        # self.term_fitness: dict[Term, torch.Tensor] = {}
        self.with_inner_evals: bool = with_inner_evals

        # self.best_term: Optional[Term] = None
        # self.best_term_outputs: Optional[torch.Tensor] = None
        # self.best_term_fitness: Optional[torch.Tensor] = None

        # self.target: torch.Tensor = target
        self.ops: dict[str, Callable] = ops
        self.listeners: list[EvalListener] = listeners

        # self.bad_fitness = torch.tensor(torch.inf, device=target.device, dtype=target.dtype)

        # self.new_listener_terms: list[Term] = []

    def get_finalizer(self):
        ''' Called on the end of solver search'''

        self.add_metrics(
            evals = self.evals,
            root_evals = self.root_evals,
            eval_calls = self.eval_calls,
            eval_cache_hits = self.eval_cache_hits,
            eval_cache_miss = self.eval_cache_miss
        )

        def finalizer():
            for output in self.new_term_outputs.values():
                del output
            self.new_term_outputs.clear()

        return finalizer

    # def is_const(
    #     self,
    #     outputs: torch.Tensor) -> Optional[torch.Tensor]:
    #     """Check if any of outputs is const or very slow function """
    #     mean = outputs.mean(dim=-1)
    #     fitness = self.fitness_fn_builder(mean)(outputs)
    #     if fitness < self.fitness_atol:
    #         return mean
    #     return None 
    
    # def is_valid(self, term: Term) -> bool:
    #     ''' All outputs are non-infinite and non-nan '''
    #     return term not in self.invalid_terms
    
    # def _get_fitness(self, term: Term) -> torch.Tensor:
    #     if term in self.term_fitness:
    #         return self.term_fitness[term]
    #     elif term in self.invalid_terms:
    #         return self.bad_fitness
    #     raise ValueError(f"Term {term} has no fitness computed")    

    def _get_binding(self, root: Term, term: Term) -> Optional[torch.Tensor]:
        term_semantics = self.semantics.get_binding(term)
        if term_semantics is not None:
            self.eval_cache_hits += 1
            return term_semantics
        if term in self.new_term_outputs:
            self.eval_cache_hits += 1
            return self.new_term_outputs[term]

        self.eval_cache_miss += 1

        return None
    
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
        
    # def _eval_loop(self, terms: list[Term]) -> list[Term]:
    #     ''' Intrenal, evaluates given terms and all produced terms by listeners'''

    #     optim_terms = []    
    #     if len(terms) > 0:
    #         optim_terms = self._eval_group(terms)

    #     while len(self.new_listener_terms) > 0:
    #         new_terms = self.new_listener_terms
    #         self.new_listener_terms.clear()
    #         self._eval_group(new_terms)
    #         self.new_listener_terms.extend((t for t in self.new_listener_terms if self._get_cached_output(t) is None))

    #     return optim_terms

    def _eval_one(self, term: Term, 
                  get_binding: Callable | None = None,
                  set_binding: Callable | None = None,
                  mode: Literal["train", "test"] = "train") -> torch.Tensor:
        ''' Interrnal, can be overriden, one term evaluation without any caching '''
        output = evaluate(term, self.ops, get_binding or self._get_binding, set_binding or self._set_binding)
        return (term, output)
    
    def eval(self, terms: list[Term] | Term) -> list[tuple[Term, torch.Tensor]] | tuple[Term, torch.Tensor]:
        ''' Evaluate given term or terms '''

        return_tensor: bool = False
        if isinstance(terms, Term):
            return_tensor = True
            terms = [terms]

        self.new_term_outputs.clear()

        term_outputs: dict[Term, torch.Tensor] = {}        
        for term in terms:
            # if self.is_cuda:
            #     cuda_stream = torch.cuda.Stream()
            #     with torch.cuda.stream(cuda_stream):
            #         output = self._eval_one(term, mode="train")
            #         term_outputs[term] = output
            # else:
            self.eval_calls += 1
            output = self._eval_one(term, mode="train")
            term_outputs[term] = output
        # if self.is_cuda:
        #     torch.cuda.synchronize()

            # return term_outputs
        
        # term_outputs, elapsed = timed(eval_timed)()
        # self.add_metrics(eval_time=elapsed)

        if self.with_inner_evals:
            new_term_outputs = self.new_term_outputs
        else:
            new_term_outputs = {v[0]:v[1] for t, v in term_outputs.items() if t in self.new_term_outputs}

        if len(new_term_outputs) > 0:
            new_terms = list(new_term_outputs.keys())
            new_outputs = list(new_term_outputs.values())
            semantics = stack_rows(new_outputs, self.dims)
            self.semantics.set_binding(new_terms, semantics)
            self.fitness.set_fitness(new_terms, semantics)
            for l in self.listeners:
                l.on_eval(new_terms, semantics)
            del semantics

        if return_tensor:
            outputs = term_outputs[terms[0]]
            return outputs
        return [term_outputs[t] for t in terms]
    
        # new_outputs = [self.new_term_outputs[t] for t in new_terms]
        # if len(new_outputs) > 0:
        #     semantics = stack_rows(new_outputs, self.target)
        #     finite_semantics_mask = torch.isfinite(semantics).all(dim=-1)  # we do not insert nans and infs
        #     (valid_ids,) = torch.where(finite_semantics_mask)
        #     (infinite_ids,) = torch.where(~finite_semantics_mask)
        #     for infinite_id in infinite_ids.tolist():
        #         invalid_term = new_terms[infinite_id]
        #         self.invalid_terms[invalid_term] = outputs[infinite_id]
        #     new_semantics = semantics[valid_ids]
        #     valid_terms = [new_terms[i] for i in valid_ids.tolist()]
        #     del semantics, finite_semantics_mask, infinite_ids, valid_ids

        #     if len(valid_terms) > 0:
        #         semantics = new_semantics

        #         new_fitness: torch.Tensor = self.fitness_fn(semantics)
        #         self.storage.insert(valid_terms, semantics)
        #         for t, f in zip(valid_terms, new_fitness):
        #             self.term_fitness[t] = f                
                
        #         for listener in self.listeners:
        #             listener_terms = listener.on_eval(valid_terms, semantics, new_fitness)
        #             if listener_terms is not None:
        #                 self.new_listener_terms.extend(listener_terms)                

        #         self._update_best_term(semantics, new_fitness)                

        #         del semantics

        # self.new_term_outputs.clear()

        # return optim_terms
    
    def get_loss_fn(self, get_binding: Callable | None = None, set_binding: Callable | None = None, no_cache: bool = False):
        ''' Differentiable function for optimization that iss aligned with fitness (nmse by default) '''
        # TODO: probably define better loss_fn - we use just (f(x) - target)^2)
        if get_binding is None: 
            new_get_binding = self._get_binding
        elif no_cache:
            new_get_binding = get_binding        
        else:
            def new_get_binding(root: Term, term: Term) -> Optional[torch.Tensor]:
                outputs = get_binding(root, term)
                if outputs is not None:
                    return outputs
                return self._get_binding(root, term)
        if set_binding is None:
            new_set_binding = self._default_set_binding
        else:
            def new_set_binding(root: Term, term: Term, value: torch.Tensor):
                set_binding(root, term, value)
                self._default_set_binding(root, term)
        def loss_fn(term: Term, *, binding = {}) -> torch.Tensor:
            outputs = evaluate(term, self.ops, new_get_binding, new_set_binding)
            return self.fitness.get_loss(outputs)
        return loss_fn

    # def eval(
    #     self,
    #     terms: Sequence[Term] | Term,
    #     # *,
    #     # return_outputs: Literal["list", "tensor"] = "list",
    #     # return_fitness: Literal["none", "list", "tensor"] = "none",
    # ) -> Evaluations:
    #     """Evaluates given terms. If terms are already in cache, results returned without affecting the metrics.
    #     Calls _eval internally, therefore could cause an avalanche of evaluations of new terms through listeners.
    #     """
    #     if isinstance(terms, Term):
    #         terms = [terms]
    #     outputs = [self._get_cached_output(term) for term in terms]
    #     eval_ids = [i for i, output in enumerate(outputs) if output is None]
    #     eval_terms = [terms[i] for i in eval_ids]
    #     if len(eval_terms) > 0:
    #         optim_terms = self._eval_loop(eval_terms)
    #         eval_outputs = [self._get_cached_output(term) for term in optim_terms]
    #         for i, eval_output in zip(eval_ids, eval_outputs):
    #             outputs[i] = eval_output
    #     output_res: list | torch.Tensor = outputs
    #     if return_outputs == "tensor":
    #         output_res = stack_rows(outputs, self.target)
    #     fitness_res: None | list | torch.Tensor = None
    #     if return_fitness != "none":
    #         fitness = [self._get_fitness(term) for term in terms]
    #         fitness_res = fitness
    #         if return_fitness == "tensor":
    #             fitness_res = torch.stack(fitness, dim=0)
    #     return Evaluations(output_res, fitness_res)   

    def test(self, get_binding: Callable[[Term], torch.Tensor]) -> torch.Tensor:
        # if self.best_term is None:
        #     raise RuntimeError("Evaluator is not fitted yet")

        # def get_binding(root: Term, term: Term) -> Optional[torch.Tensor]:
        #     if isinstance(term, Variable):
        #         return var_bindings[term.var_id]
        #     if isinstance(term, Value):
        #         # return self.const_binding[term.value]
        #         return term.value
        #     return None

        def set_binding(*_):
            pass

        _, output = self._eval_one(self.fitness.best_term, get_binding, set_binding, mode="test")
        return output      

    def get_iter_metrics(self):
        return {
            'iter_evals': [self.evals],
            'iter_root_evals': [self.root_evals],
            'iter_eval_calls': [self.eval_calls],
            'iter_eval_cache_hits': [self.eval_cache_hits],
            'iter_eval_cache_miss': [self.eval_cache_miss],
        }              

class OptimEvaluator(DefaultEvaluator):
    ''' Perform optimization before evaluation '''

    def __init__(self, *,
                 optimizer: Optimizer,
                 **kwargs):
        super().__init__(**kwargs)
        self.optimizer = optimizer
        self.term_mapping: dict[Term, Term] = {} # term to optimized term mapping

    def _eval_one(self, term: Term, 
                  get_binding: Callable | None = None,
                  set_binding: Callable | None = None,
                  mode: Literal["train", "test"] = "train") -> torch.Tensor:
        ''' Optimize and then evaluate the term '''        
        if term in self.term_mapping:
            optim_term = self.term_mapping[term]
        elif mode == "test":
            optim_term = term
        else:
            optim_term = self.optimizer.optimize(term)
            self.term_mapping[term] = optim_term
        res = super()._eval_one(optim_term, get_binding, set_binding, mode)
        return res

    def get_loss_fn(self, get_binding: Callable | None = None, set_binding: Callable | None = None, no_cache: bool = False):
        return self.evaluator.get_loss_fn(get_binding=get_binding, set_binding=set_binding, no_cache = no_cache)