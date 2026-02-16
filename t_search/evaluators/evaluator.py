''' Base interface for syntax Evaluators'''

from collections.abc import Callable
from typing import Literal, Optional, Sequence

import torch

from t_search.base import ServiceBase
from t_search.evaluators.fitness import Fitness

from t_search.evaluators.semantics import Semantics
from t_search.operators.operator import Operator
from t_search.syntax.evaluation import evaluate
from t_search.syntax.term import Term, Value, Variable
from t_search.utils import EvSearchTermination, stack_rows

class Evaluator(Operator, ServiceBase):
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
                    device: torch.device,
                    measure_loss_each_n: int = 0,
                    measure_max_len: int = 100
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
        self.optim_evals: int = 0 # to track spent time on optimization
        self.eval_calls: int = 0
        # self.eval_fn = eval_fn
        self.add_metrics = add_metrics
        self.is_cuda = device.type == 'cuda'
        self.evals_simple: int = 0 # count of 1 node evals (consts, free vars)
        self.eval_cache_hits: int = 0
        self.eval_cache_miss: int = 0
        self.measure_loss_each_n = measure_loss_each_n
        self.measure_max_len = measure_max_len
        self.best_loss: float = float('inf')
        self.loss_trace: list[float] = []

        self.new_term_outputs: dict[Term, torch.Tensor] = {}        
        # self.const_term_outputs: dict[Term, torch.Tensor] = {}

        # self.term_fitness: dict[Term, torch.Tensor] = {}
        self.with_inner_evals: bool = with_inner_evals

        # self.best_term: Optional[Term] = None
        # self.best_term_outputs: Optional[torch.Tensor] = None
        # self.best_term_fitness: Optional[torch.Tensor] = None

        # self.target: torch.Tensor = target
        self.ops: dict[str, Callable] = ops
        # self.listeners: list[EvalListener] = []

        # self.bad_fitness = torch.tensor(torch.inf, device=target.device, dtype=target.dtype)

        # self.new_listener_terms: list[Term] = []

    # def add_listeners(self, listeners: list[EvalListener]):
    #     self.listeners.extend(listeners)

    # def add_initial_terms(self, terms: list[Term], semantics: torch.Tensor):
    #     ''' Add initial terms and their semantics to evaluator storage '''
    #     self.semantics.set_binding(terms, semantics)
    #     self.fitness.set_fitness(terms, semantics)
    #     valid_terms = []
    #     valid_semantic_list = []
    #     for term, sem in zip(terms, semantics):
    #         if self.semantics.is_valid(term):
    #             valid_terms.append(term)
    #             valid_semantic_list.append(sem)
    #     if len(valid_terms) > 0:
    #         for l in self.listeners:
    #             l.on_eval(valid_terms, valid_semantic_list)

    def get_finalizer(self):
        ''' Called on the end of solver search'''

        self.add_metrics(
            evals = self.evals,
            optim_evals = self.optim_evals,
            root_evals = self.root_evals,
            eval_calls = self.eval_calls,
            eval_cache_hits = self.eval_cache_hits,
            eval_cache_miss = self.eval_cache_miss,
            loss_trace = self.loss_trace,
            loss_each_n = self.measure_loss_each_n
        )

        def finalizer():
            for output in self.new_term_outputs.values():
                del output
            self.new_term_outputs.clear()

        return finalizer

    def _get_binding(self, root: Term, term: Term) -> Optional[torch.Tensor]:
        term_semantics = self.semantics.get_binding(term)
        if term_semantics is not None:
            if isinstance(term, (Variable, Value)):
                self.evals_simple += 1
            self.eval_cache_hits += 1
            return term_semantics
        if term in self.new_term_outputs:
            self.eval_cache_hits += 1
            return self.new_term_outputs[term]

        self.eval_cache_miss += 1

        return None
    
    def _default_set_binding(self, root: Term, term: Term, value: torch.Tensor, 
                             under_optim: bool = False):
        self.evals += 1
        if under_optim:
            self.optim_evals += 1
        if root == term:
            self.root_evals += 1
        if len(self.loss_trace) < self.measure_max_len:
            if not under_optim:
                cur_loss = self.fitness.get_loss(value).mean(dim=-1).min().item()
                self.best_loss = min(self.best_loss, cur_loss)
            if self.evals % self.measure_loss_each_n == 0:
                self.loss_trace.append(self.best_loss)
        if self.evals >= self.max_evals:
            raise EvSearchTermination("MAX_EVAL")
        if self.root_evals >= self.max_root_evals:
            raise EvSearchTermination("MAX_ROOT_EVAL")
        
    def add_external_optim_eval_cnt(self):
        self.evals += 1
        self.optim_evals += 1
        if len(self.loss_trace) < self.measure_max_len:
            if self.evals % self.measure_loss_each_n == 0:
                self.loss_trace.append(self.best_loss)
        if self.evals >= self.max_evals:
            raise EvSearchTermination("MAX_EVAL")   
    
    def _set_binding(self, root: Term, term: Term, value: torch.Tensor):
        self._default_set_binding(root, term, value)
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
        return output
    
    def eval(self, terms: list[Term] | Term, return_type: Literal["tensor", "list"] = "list",
                ensure_dims: bool = False) -> torch.Tensor:
        ''' Evaluate given term or terms '''

        if isinstance(terms, Term):
            return_tensor = True
            terms = [terms]

        self.new_term_outputs.clear()

        term_outputs: dict[Term, torch.Tensor] = {}        
        for term in terms:
            self.eval_calls += 1
            output = self._eval_one(term, mode="train")
            term_outputs[term] = output

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
            # valid_terms = []
            # valid_semantic_list = []
            # for term, sem in zip(new_terms, semantics):
            #     if self.semantics.is_valid(term):
            #         valid_terms.append(term)
            #         valid_semantic_list.append(sem)
            # if len(valid_terms) > 0:    
            #     for l in self.listeners:
            #         l.on_eval(valid_terms, valid_semantic_list)
            del semantics

        if len(terms) == 1:
            s = term_outputs[terms[0]]
            if ensure_dims and s.numel() == 1:
                s = s.expand(self.dims)
            return s
        results = [term_outputs[t] for t in terms]
        if return_type == "tensor":
            res = stack_rows(results, self.dims)
            return res
        elif ensure_dims:
            final_results = []
            for r in results:
                if r.numel() == 1:
                    final_results.append(r.expand(self.dims))
                else:
                    final_results.append(r)
            return final_results
        return results
    
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
            def new_set_binding(root: Term, term: Term, value: torch.Tensor):
                self._default_set_binding(root, term, value, under_optim=True)
        else:
            def new_set_binding(root: Term, term: Term, value: torch.Tensor):
                set_binding(root, term, value)
                self._default_set_binding(root, term, value, under_optim=True)
        def loss_fn(term: Term) -> torch.Tensor:
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

        if self.fitness.best_term is None:
            raise RuntimeError("Evaluator is not fitted yet")

        _, output = self._eval_one(self.fitness.best_term, get_binding, set_binding, mode="test")
        return output      

    def get_iter_metrics(self):
        return {
            'iter_evals': [self.evals],
            'iter_optim_evals': [self.optim_evals],
            'iter_root_evals': [self.root_evals],
            'iter_evals_simple': [self.evals_simple],
            'iter_eval_calls': [self.eval_calls],
            'iter_eval_cache_hits': [self.eval_cache_hits],
            'iter_eval_cache_miss': [self.eval_cache_miss],
        }              
    
    def __call__(self, population: Sequence[Term]) -> Sequence[Term]:
        ''' Evaluate given population of terms and return them'''
        self.eval(population)
        return population    