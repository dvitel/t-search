"""Population based evolutionary loop and default operators, Koza style GP.
Operators:
    1. Initialization: ramped-half-and-half
    2. Selection: tournament
    3. Crossover: one-point subtree
    4. Mutation: one-point subtree
"""

from functools import partial
from time import perf_counter
from typing import Any, Callable, Literal, NamedTuple, Optional, Sequence, Type

import numpy as np
import torch
from operators import RHH, RPM, RPX, TS, Initialization, Operator, TermsListener
from sklearn.base import BaseEstimator, RegressorMixin

# from mutation import CO, Dedupl, Mutation, PO, RPX, RPM, Reduce
from .term import (
    Builder,
    Builders,
    Op,
    Term,
    TermGenContext,
    TermPos,
    UnifyBindings,
    Value,
    Variable,
    evaluate,
    get_counts,
    get_depth,
    get_fn_arity,
    get_pos_constraints,
    get_pos_sibling_counts,
    get_positions,
    is_valid,
    match_root,
    parse_term,
    replace_pos,
    replace_pos_protected,
)
from .torch_alg import nmse_loss_builder
from .util import stack_rows

GPSolverStatus = Literal["INIT", "MAX_GEN", "MAX_EVAL", "MAX_ROOT_EVAL", "SOLVED"]


class TermEvals(NamedTuple):
    outputs: list[torch.Tensor] | torch.Tensor
    fitness: None | list[torch.Tensor] | torch.Tensor


class EvSearchTermination(Exception):
    """Reaching maximum of evals, gens, ops etc"""

    def __init__(self, status: GPSolverStatus, *args):
        super().__init__(*args)
        self.status = status


def fit_0(
    fitness: torch.Tensor,
    prev_best_fitness: Optional[torch.Tensor],
    rtol=1e-04,
    atol=1e-03,
) -> tuple[int | None, bool]:
    best_fitness, best_id = torch.min(fitness, dim=0)
    best_found = False
    if prev_best_fitness is not None and (prev_best_fitness < best_fitness):
        return None, False
    zero = torch.zeros_like(best_fitness)
    if torch.isclose(best_fitness, zero, rtol=rtol, atol=atol):
        best_found = True
    best_id_value = best_id.item()
    return int(best_id_value), best_found


def timed(fn: Callable, key: str, metrics: dict) -> Callable:
    """Decorator to time function execution"""

    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        try:
            result = fn(*args, **kwargs)
        finally:
            elapsed_time = round((perf_counter() - start_time) * 1000)
            metrics[key] = metrics.get(key, 0) + elapsed_time
        return result

    return wrapper


class GPSolver(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        ops: dict[str, Callable],
        fitness_fn_builder: Callable = nmse_loss_builder,
        fit_condition=partial(fit_0, rtol=1e-04, atol=1e-03),
        init: Initialization = RHH(),
        eval_fn=evaluate,
        pipeline: list["Operator"] = [TS(), RPM(), RPX()],
        ops_counts: dict[str, tuple[int, int]] = {},
        forbid_patterns: list[str] = [],
        # next is more optimized
        inner_ops_max_counts: dict[str, dict[str, int]] = {},
        immediate_arg_limits: dict[str, dict[str, int]] = {},
        prohibit_ops_on_consts_only: bool = True,
        # commutative_ops: list[str] = [], # by all args
        max_term_depth=17,
        min_consts: int = 0,
        max_consts: int = 5,  # 0 to disable consts in terms
        min_vars: int = 1,
        max_vars: int = 10,  # max number of free variables
        max_ops: dict[str, int] = {},
        max_gen: int = 100,
        max_root_evals: int = 100_000,
        max_evals: int = 1_000_000,
        pop_size: int = 1000,
        with_inner_evals: bool = False,
        cache_term_props: bool = True,
        # cache_terms: bool = True,
        # cache_evals: bool = True, # outputs and fitness
        # compute_output_range = True,
        const_range: Optional[tuple[float, float]] = None,  # if not set, computed from X, y
        rtol=1e-04,
        atol=1e-03,  # NOTE: these are for semantic/outputs comparison,
        # not for fitness, see fit_0
        rnd_seed: Optional[int] = None,
        torch_rnd_seed: Optional[int] = None,
        device="cpu",
        dtype=torch.float32,
    ):

        self.ops = ops
        self.ops_counts = ops_counts

        self.max_term_depth = max_term_depth
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.min_consts = min_consts
        self.max_consts = max_consts
        self.max_ops = max_ops
        self.forbid_patterns = forbid_patterns
        self.fpatterns = []
        if len(self.forbid_patterns) > 0:
            self.match_cache: dict[tuple, UnifyBindings] = {}
            for p in self.forbid_patterns:
                t, i = parse_term(p)
                assert len(p) == i, f"Invalid pattern: {p}"
                self.fpatterns.append(t)

        # self.const_binding: list[torch.Tensor] = []
        self.const_range: torch.Tensor | None = None  # detected from y
        if const_range is not None:
            self.const_range = torch.tensor(
                const_range,
                dtype=dtype,
                device=device,
            )

        self.fitness_fn_builder = fitness_fn_builder
        self.fit_condition = fit_condition
        self.init = init
        self.eval_fn = eval_fn
        self.pipeline = pipeline
        self.max_gen = max_gen
        self.max_root_evals = max_root_evals
        self.max_evals = max_evals
        self.pop_size = pop_size
        self.with_inner_evals = with_inner_evals
        self.cache_term_props = cache_term_props
        # self.cache_terms = cache_terms
        # self.cache_evals = cache_evals
        self.rtol = rtol
        self.atol = atol
        self.device = device
        self.dtype = dtype
        # self.output_range = None
        # self.compute_output_range = compute_output_range
        self.prohibit_ops_on_consts_only = prohibit_ops_on_consts_only
        self.inner_ops_max_counts = inner_ops_max_counts
        self.immediate_arg_limits = immediate_arg_limits
        # self.commutative_ops = set(commutative_ops)

        self.rnd: np.random.Generator = np.random.default_rng(rnd_seed)

        if torch_rnd_seed is None:
            self.torch_gen = None
        else:
            self.torch_gen = torch.Generator(device=device)
            self.torch_gen.manual_seed(torch_rnd_seed)

        # next are runtime fields and caches that works across fit calls
        self.target: torch.Tensor = torch.empty(
            0,
            device=self.device,
            dtype=self.dtype,
        )
        self.term_outputs: dict[Term, torch.Tensor] = {}
        self.new_term_outputs: dict[Term, torch.Tensor] = {}
        # terms with nans or infs in output
        self.invalid_term_outputs: dict[Term, torch.Tensor] = {}
        # self.const_term_outputs: dict[Term, torch.Tensor] = {}
        self.term_fitness: dict[Term, torch.Tensor] = {}
        self.pos_cache: dict[Term, list[TermPos]] = {}
        self.pos_context_cache: dict[
            Term,
            dict[tuple[Term, int], TermGenContext],
        ] = {}
        self.depth_cache: dict[Term, int] = {}
        self.counts_cache: dict[Term, np.ndarray] = {}

        self.best_term: Optional[Term] = None
        self.best_outputs: Optional[torch.Tensor] = None
        self.best_fitness: Optional[torch.Tensor] = None
        self.gen: int = 0
        self.evals: int = 0
        self.root_evals: int = 0
        self.metrics: dict[str, Any] = {}
        self.gen_metrics: dict[str, Any] = {}
        self.is_fitted_: bool = False
        self.status: GPSolverStatus = "INIT"
        self.start_time: float = 0
        self.const_id: int = 0
        self.const_tape: torch.Tensor | None = None
        self.term_listeners: list[TermsListener] = []

        self.op_builders: dict[str, Builder] = {}
        for op_id, op_fn in self.ops.items():
            op_arity = get_fn_arity(op_fn)
            max_count = None
            if op_id in self.max_ops:
                max_count = self.max_ops[op_id]
            op_builder = Builder(
                op_id,
                self._alloc_op_builder(op_id),
                op_arity,
                max_count=max_count,
            )
            # commutative = op_id in self.commutative_ops)
            if op_id in self.ops_counts:
                op_min_count, op_max_count = self.ops_counts[op_id]
                op_builder.min_count = op_min_count
                op_builder.max_count = op_max_count

            self.op_builders[op_id] = op_builder

    def _reset_state(
        self,
        free_vars: Sequence | torch.Tensor,
        target: Sequence | torch.Tensor,
    ):
        """Called before each fit"""

        # reset caches
        self.vars = {}
        self.var_binding = {}

        self.syntax.clear()
        for output in self.term_outputs.values():
            del output
        self.term_outputs.clear()
        for output in self.new_term_outputs.values():
            del output
        self.new_term_outputs.clear()
        for output in self.invalid_term_outputs.values():
            del output
        self.invalid_term_outputs.clear()
        for fitness in self.term_fitness.values():
            del fitness
        self.term_fitness.clear()
        self.pos_cache.clear()
        self.pos_context_cache.clear()
        self.counts_cache.clear()
        self.depth_cache.clear()

        self.best_term = None
        self.best_outputs = None
        self.best_fitness = None
        self.gen = 0
        self.evals = 0
        self.root_evals = 0
        self.metrics.clear()
        self.status = "INIT"
        self.start_time = perf_counter()
        self.gen_metrics = {}
        self.metrics["gens"] = [self.gen_metrics]
        self.is_fitted_ = False
        builders: dict[Type | str, Builder] = {}

        self.const_builder = None
        if self.max_consts > 0:
            self.const_builder = Builder("C", self._alloc_const, 0, self.min_consts, self.max_consts)
            builders[Value] = self.const_builder

        self.var_builder = None
        if free_vars is not None and len(free_vars) > 0 and (self.max_vars > 0):
            vars, var_binding = self._get_vars(free_vars)
            self.var_binding = var_binding
            self.vars = {v.var_id: v for v in vars}
            self.var_builder = Builder("x", self._alloc_var, 0, self.min_vars, self.max_vars)
            builders[Variable] = self.var_builder

        for op_id, op_builder in self.op_builders.items():
            builders[op_id] = op_builder

        def get_term_builder(term: Term):
            if isinstance(term, Op):
                builder = builders[term.op_id]
            if isinstance(term, Variable):
                builder = builders[Variable]
            if isinstance(term, Value):
                builder = builders[Value]
            return builder

        self.builders = Builders(list(builders.values()), get_term_builder)

        arg_limits = {}
        if self.prohibit_ops_on_consts_only:
            for b in self.op_builders.values():
                arg_limits[b] = {builders[Value]: b.arity() - 1}

        for op_id, op_dict in self.immediate_arg_limits.items():
            if op_id not in self.op_builders:
                continue
            b = self.op_builders[op_id]
            if b not in arg_limits:
                arg_limits[b] = {}
            for inner_op_id, limit in op_dict.items():
                if inner_op_id not in self.op_builders:
                    raise ValueError(f"Inner operator {inner_op_id} " "not found in op_builders")
                arg_limits[b][self.op_builders[inner_op_id]] = limit

        self.builders.limit_args(arg_limits)

        context_limits = {}
        for op_id, op_limits in self.inner_ops_max_counts.items():
            if op_id not in self.op_builders:
                continue
            context_limits[self.op_builders[op_id]] = {
                self.op_builders[inner_op_id]: cnt for inner_op_id, cnt in op_limits.items()
            }

        self.builders.limit_context(context_limits)

        if not torch.is_tensor(target):
            self.target = torch.tensor(
                target,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.target = target.to(device=self.device, dtype=self.dtype)

        if self.const_range is None:
            min_value = self.target.min()
            max_value = self.target.max()
            if torch.isclose(
                min_value,
                max_value,
                rtol=self.rtol,
                atol=self.atol,
            ):
                min_value = min_value - 0.1
                max_value = max_value + 0.1
            dist = max_value - min_value
            min_value = min_value - 0.1 * dist
            max_value = max_value + 0.1 * dist
            self.const_range = torch.tensor([min_value, max_value], dtype=self.dtype, device=self.device)
            if not torch.is_tensor(free_vars):
                free_vars = torch.stack([fv for fv in free_vars], dim=0)
            min_fv = torch.min(free_vars)
            max_fv = torch.max(free_vars)
            self.const_range[0] = torch.minimum(self.const_range[0], min_fv)
            self.const_range[1] = torch.maximum(self.const_range[1], max_fv)

        self.fitness_fn: Callable[[torch.Tensor], torch.Tensor] = self.fitness_fn_builder(self.target)

        # self.output_range = torch.stack([self.target, self.target], dim=0)
        # abs_target = torch.abs(self.target)
        # self.output_range[0] -= 0.1 * abs_target
        # self.output_range[1] += 0.1 * abs_target
        # del abs_target

        for op in self.pipeline:
            op.op_init(self)

        self.term_listeners = [op for op in self.pipeline if isinstance(op, TermsListener)]

        for t_listener in self.term_listeners:
            for x in self.vars.values():
                binding = self.var_binding[x.var_id]
                t_listener.register_terms(self, [x], binding.unsqueeze(0))
        pass

    def _get_vars(self, free_vars):
        vars = []
        var_binding = {}
        for i, xi in enumerate(free_vars):
            v = Variable(f"x{i}")
            if not torch.is_tensor(xi):
                fv = torch.tensor(xi, dtype=self.dtype, device=self.device)
            else:
                fv = xi.to(device=self.device, dtype=self.dtype)
            vars.append(v)
            var_binding[v.var_id] = fv
        return vars, var_binding

    def _alloc_var(self, *, var_id: Optional[str] = None) -> Variable:
        if var_id is not None:
            var = self.vars.get(var_id, None)
            if var is not None:
                return var
        var = self.rnd.choice(list(self.vars.values()))
        return var

    def _alloc_const(self, *, value: Optional[float | torch.Tensor] = None) -> Value:
        """Should we random sample of try some grid? Anyway we tune"""
        if value is not None:  # const value provided - no alloc of consts
            if not torch.is_tensor(value):
                value = torch.tensor(value, dtype=self.dtype, device=self.device)
            return Value(value)
        if self.const_id == 0 or self.const_id >= self.pop_size:
            del self.const_tape
            self.const_tape = torch.rand(
                self.pop_size,
                device=self.device,
                dtype=self.dtype,
                generator=self.torch_gen,
            )
            if self.const_range is not None:
                dist = self.const_range[1] - self.const_range[0]
                self.const_tape *= dist
                self.const_tape += self.const_range[0]
        assert self.const_tape is not None
        value = self.const_tape[self.const_id]
        self.const_id += 1
        # const_id = len(self.const_binding)
        # self.const_binding.append(value)
        # return Value(const_id)
        self.metrics["consts"] = self.metrics.get("consts", 0) + 1
        return Value(value)

    def report_evals(self, num_evals: int, num_root_evals: int):
        self.evals += num_evals
        self.root_evals += num_root_evals
        if self.evals > self.max_evals:
            raise EvSearchTermination("MAX_EVAL", "Maximum number of evaluations reached")
        if self.root_evals > self.max_root_evals:
            raise EvSearchTermination("MAX_ROOT_EVAL", "Maximum number of root evaluations reached")

    def _eval(self, terms: list[Term]):

        self.new_term_outputs.clear()

        outputs = []
        for term in terms:
            output = self.eval_fn(term, self.ops, self._get_binding, self._set_binding)
            outputs.append(output)

        new_terms = [t for t in terms if t in self.new_term_outputs]
        if self.with_inner_evals:
            new_terms = list(self.new_term_outputs.keys())

        # assert len(terms) > 0, "No terms to update fitness for"

        outputs = [self.new_term_outputs[t] for t in new_terms]
        if len(outputs) > 0:
            semantics = stack_rows(outputs, target_size=self.target.shape[0])
            finite_semantics_mask = torch.isfinite(semantics).all(dim=-1)  # we do not insert nans and infs
            (valid_ids,) = torch.where(finite_semantics_mask)
            (infinite_ids,) = torch.where(~finite_semantics_mask)
            for infinite_id in infinite_ids.tolist():
                invalid_term = new_terms[infinite_id]
                self.invalid_term_outputs[invalid_term] = outputs[infinite_id]
            new_semantics = semantics[valid_ids]
            valid_terms = [new_terms[i] for i in valid_ids.tolist()]
            del semantics, finite_semantics_mask, infinite_ids, valid_ids

            # check for const semantics
            # if len(valid_terms) > 0:
            #     semantics = new_semantics
            #     semantics_mean = semantics.mean(dim=-1, keepdim=True)
            #     const_el_mask = torch.isclose(semantics, semantics_mean, rtol = self.rtol, atol = self.atol)
            #     const_mask = const_el_mask.all(dim=-1)
            #     const_ids, = torch.where(const_mask)
            #     nonconst_ids, = torch.where(~const_mask)
            #     for const_id in const_ids.tolist():
            #         const_term = valid_terms[const_id]
            #         self.const_term_outputs[const_term] = semantics_mean[const_id]
            #     new_semantics = semantics[nonconst_ids]
            #     valid_terms = [valid_terms[i] for i in nonconst_ids.tolist()]
            #     del semantics, semantics_mean, const_el_mask, const_mask, const_ids, nonconst_ids

            if len(valid_terms) > 0:
                semantics = new_semantics
                for t_listener in self.term_listeners:
                    t_listener.register_terms(self, valid_terms, semantics)

                # if self.compute_output_range:
                #     finite_predictions_mask = torch.isfinite(predictions).all(dim=-1)
                #     if torch.any(finite_predictions_mask):
                #         finite_predictions = predictions[finite_predictions_mask]
                #         min_outputs = torch.min(finite_predictions, dim=0).values
                #         max_outputs = torch.max(finite_predictions, dim=0).values
                #         torch.minimum(self.output_range[0], min_outputs, out=self.output_range[0])
                #         torch.maximum(self.output_range[1], max_outputs, out=self.output_range[1])
                new_fitness: torch.Tensor = self.fitness_fn(semantics)
                for t, o, f in zip(valid_terms, semantics, new_fitness):
                    self.term_outputs[t] = o.clone()
                    self.term_fitness[t] = f
                del semantics

                best_id, best_found = self.fit_condition(new_fitness, self.best_fitness)
                if best_id is not None:
                    self._set_best_term(
                        valid_terms[best_id],
                        outputs[best_id].clone(),
                        new_fitness[best_id].clone(),
                    )
                    if best_found:
                        raise (EvSearchTermination("SOLVED"))

        self.new_term_outputs.clear()

        return

    def _breed(self, population: list[Term]) -> list[Term]:
        """Pipeline that mutates parents and then applies crossover on pairs. One-point operations"""

        # caches

        if not self.cache_term_props:
            self.pos_cache.clear()
            self.counts_cache.clear()
            self.pos_context_cache.clear()
            self.depth_cache.clear()

        # validation 1
        # for term in population:
        #     poss = get_positions(term, {})
        #     for pos in poss:
        #         get_pos_constraints(pos, self.builders, {}, {})
        #     pass

        children = population

        for operator in self.pipeline:
            children = timed(operator, f"{operator.name}_time", self.gen_metrics)(self, children)
            if len(operator.metrics) > 0:
                self.gen_metrics[operator.name] = operator.metrics

        return children

    def _validate_patterns(self, term: Term) -> bool:
        for fpattern in self.fpatterns:
            match = match_root(term, fpattern, prev_matches=self.match_cache)
            if match is not None:
                return False
        return True

    def _alloc_op_builder(self, op_id: str) -> Callable:

        hit_key = "syntax_hit"
        miss_key = "syntax_miss"

        # if self.cache_terms:
        def _alloc_op(*args):
            signature = (op_id, *args)
            if signature in self.syntax:
                key = hit_key
                term = self.syntax[signature]
            else:
                key = miss_key
                term = Op(op_id, args)
                self.syntax[signature] = term
            if not self._validate_patterns(term):
                self.syntax.pop(signature, None)
                return None
            self.metrics[key] = self.metrics.get(key, 0) + 1
            if term in self.invalid_term_outputs:
                self.metrics["syntax_invalid"] = self.metrics.get("syntax_invalid", 0) + 1
                return None  # do not output known invalid terms
            # elif term in self.const_term_outputs:
            #     self.metrics["syntax_const"] = self.metrics.get("syntax_const", 0) + 1
            #     # return Value(self.const_term_outputs[term]) # return const value
            #     # return None
            #     pass
            #     # NOTE: returning value could ruin constraints. Instead, we disallow constant terms because constant leaf could be used instead.
            #     # TODO: separate operator that transforms terms to simple forms with removed constants
            return term

        # else:
        #     def _alloc_op(*args):
        #         self.metrics[miss_key] = self.metrics.get(miss_key, 0) + 1
        #         term = Op(op_id, args)
        #         if self.validate_term(term):
        #             return term
        #         return None

        return _alloc_op

    def _get_cached_output(self, term: Term) -> Optional[torch.Tensor]:
        if isinstance(term, Variable):
            return self.var_binding[term.var_id]
        if isinstance(term, Value):
            # return self.const_binding[term.value]
            return term.value
        if term in self.term_outputs:
            semantics = self.term_outputs[term]
            return semantics
        if term in self.new_term_outputs:
            return self.new_term_outputs[term]
        if term in self.invalid_term_outputs:
            return self.invalid_term_outputs[term]
        # if term in self.const_term_outputs:
        #     return self.const_term_outputs[term]
        return None

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
        if self.evals == self.max_evals:
            raise EvSearchTermination("MAX_EVAL")
        if self.root_evals == self.max_root_evals:
            raise EvSearchTermination("MAX_ROOT_EVAL")
        self.new_term_outputs[term] = value

    def _checkpoint_metrics(self):
        self.gen_metrics = {}
        self.metrics["gens"].append(self.gen_metrics)

    def _timed_eval(self, population):
        return timed(self._eval, "eval_time", self.gen_metrics)(population)

    # NOTE: this is public interface to eval from operators.
    def eval(
        self,
        terms: Sequence[Term] | Term,
        *,
        return_outputs: Literal["list", "tensor"] = "list",
        return_fitness: Literal["none", "list", "tensor"] = "none",
    ) -> TermEvals:
        """Evaluates given terms. If terms are already in cache, results returned without affecting the metrics.
        Calls _eval internally, therefore could cause an avalanche of evaluations of new terms through listeners.
        """
        if isinstance(terms, Term):
            terms = [terms]
        outputs = [self._get_cached_output(term) for term in terms]
        eval_ids = [i for i, output in enumerate(outputs) if output is None]
        eval_terms = [terms[i] for i in eval_ids]
        if len(eval_terms) > 0:
            self._timed_eval(eval_terms)
            eval_outputs = [self._get_cached_output(term) for term in eval_terms]
            for i, eval_output in zip(eval_ids, eval_outputs):
                outputs[i] = eval_output
        output_res: list | torch.Tensor = outputs
        if return_outputs == "tensor":
            output_res = stack_rows(outputs, target_size=self.target.shape[0])
        fitness_res: None | list | torch.Tensor = None
        if return_fitness != "none":
            fitness = [self.term_fitness[term] for term in terms]
            fitness_res = fitness
            if return_fitness == "tensor":
                fitness_res = torch.stack(fitness, dim=0)
        return TermEvals(outputs_res, fitness_res)

    def _loop(self):
        population = timed(self.init, "init_time", self.gen_metrics)(self, self.pop_size)
        self.gen_metrics.update(self.init.metrics)
        self._timed_eval(population)
        self._checkpoint_metrics()
        while self.gen < self.max_gen and self.evals < self.max_evals and self.root_evals < self.max_root_evals:
            population = self._breed(population)
            self._timed_eval(population)
            self.gen += 1
            self._checkpoint_metrics()

    def _add_final_metrics(self):
        self.metrics["gen"] = self.gen
        self.metrics["evals"] = self.evals
        self.metrics["root_evals"] = self.root_evals
        self.metrics["final_time"] = round((perf_counter() - self.start_time) * 1000)
        self.metrics["status"] = self.status
        if self.best_term is not None:
            self.metrics["solution"] = self.best_term

    def find_any_const(
        self,
        outputs: torch.Tensor,
        atol: float | None = None,
        rtol: float | None = None,
    ) -> Optional[torch.Tensor]:
        """Check if output is const or very slow function"""
        means = outputs.mean(dim=-1, keepdim=True)
        close_el_mask = torch.isclose(outputs, means, rtol=rtol or self.rtol, atol=atol or self.atol)
        close_mask = close_el_mask.all(dim=-1)
        (const_ids,) = torch.where(close_mask)
        if len(const_ids) > 0:
            return means[const_ids[0], 0]
        return None

    def find_any_var(self, outputs: torch.Tensor) -> Optional[Variable]:
        for x in self.vars.values():
            x_binding = self.var_binding[x.var_id]
            close_el_mask = torch.isclose(outputs, x_binding, rtol=self.rtol, atol=self.atol)
            cur_close_mask = close_el_mask.all(dim=-1)
            (cur_close_ids,) = torch.where(cur_close_mask)
            if len(cur_close_ids) > 0:
                return x
        return None

    def get_depth(self, term: Term) -> int:
        term_depth = get_depth(term, self.depth_cache)
        return term_depth

    def get_positions(self, term: Term) -> list[TermPos]:
        term_pos = get_positions(term, self.pos_cache)
        return term_pos

    def get_gen_constraints(self, term: Term, pos: TermPos) -> tuple[TermGenContext, np.ndarray]:
        start_context = get_pos_constraints(
            pos,
            self.builders,
            self.counts_cache,
            self.pos_context_cache.setdefault(term, {}),
        )
        arg_counts = get_pos_sibling_counts(pos, self.builders)
        return start_context, arg_counts

    def replace_position(self, term: Term, pos: TermPos, new_subterm: Term, with_validation=True) -> Optional[Term]:
        if with_validation:
            child = replace_pos_protected(
                pos,
                new_subterm,
                self.builders,
                depth_cache=self.depth_cache,
                counts_cache=self.counts_cache,
                pos_context_cache=self.pos_context_cache.setdefault(term, {}),
                max_term_depth=self.max_term_depth,
            )
        else:
            child = replace_pos(pos, new_subterm, self.builders)
        return child

    def is_valid(self, term: Term) -> bool:
        term_is_valid = is_valid(term, builders=self.builders, counts_cache=self.counts_cache)
        pattern_valid = self._validate_patterns(term)
        return term_is_valid and pattern_valid

    def _set_best_term(self, best_term: Term, best_outputs: torch.Tensor, best_fitness: torch.Tensor):
        self.best_term = best_term
        self.best_outputs = best_outputs
        self.best_fitness = best_fitness
        best_fitness = best_fitness.unsqueeze(-1) if len(best_fitness.shape) == 0 else best_fitness
        for fi, fv in enumerate(best_fitness):
            fk = f"best_fitness_{fi}"
            self.gen_metrics[fk] = fv.item()

        best_depth = get_depth(best_term, self.depth_cache)
        best_counts = get_counts(best_term, self.builders, self.counts_cache)
        best_size = best_counts.sum().item()
        self.gen_metrics["best_term_depth"] = best_depth
        self.gen_metrics["best_term_size"] = best_size
        self.gen_metrics["best_counts"] = best_counts.tolist()
        self.gen_metrics["best_term"] = str(best_term)

    def _check_trivial(self):
        const_val = self.find_any_const(self.target.unsqueeze(0))
        if const_val is not None:  # NOTE: or torch.any ??? config option
            best_term = Value(const_val)  # len(self.const_binding))
            best_fitness = torch.tensor(0, device=self.device, dtype=self.dtype)
            best_outputs = const_val
            self._set_best_term(best_term, best_outputs, best_fitness)
            self.status = "SOLVED"
            return True
        x = self.find_any_var(self.target.unsqueeze(0))
        if x is not None:
            best_term = x
            best_fitness = torch.tensor(0, device=self.device, dtype=self.dtype)
            best_outputs = self.var_binding[x.var_id]
            self._set_best_term(best_term, best_outputs, best_fitness)
            self.status = "SOLVED"
            return True
        return False

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> "GPSolver":
        """
        Fit the solver to the data.

        Args:
            X (array-like): Input features.
            y (array-like): Target labels.

        Returns:
            self: Returns the instance itself.
        """
        self._reset_state(free_vars=X, target=y)
        if self._check_trivial():
            return self
        try:
            self._loop()
        except EvSearchTermination as e:
            self.status = e.status
        self.is_fitted_ = True
        if self.status == "INIT":
            self.status = "MAX_GEN"
        self._add_final_metrics()
        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Predict using the trained solver.

        Args:
            X (array-like): Input features.

        Returns:
            array-like: Predicted values.
        """
        if not self.is_fitted_ or self.best_term is None:
            raise RuntimeError("Solver is not fitted yet")

        _, var_binding = self._get_vars(X)

        def get_binding(root: Term, term: Term) -> Optional[torch.Tensor]:
            if isinstance(term, Variable):
                return var_binding[term.var_id]
            if isinstance(term, Value):
                # return self.const_binding[term.value]
                return term.value
            return None

        def set_binding(*_):
            pass

        output = self.eval_fn(self.best_term, self.ops, get_binding, set_binding)
        if output is None:
            raise RuntimeError("Evaluation of the best term returned None, not all terminals may be bound")
        output_numpy = output.cpu().numpy()
        return output_numpy


# NOTE: on metrics:
#       1. In contrast to cde-search, we do not collect semantic and syntactic diversity measures of population
#          For algebraic domain provided notions of diversity could be noninformative and should be reconsidered.
#       2. Also, we do not collect children "betterness" as it would require preservation of parent evaluations
#          till moment of children evaluation. To avoid logic compication we decided to avoid this. Also algebraic domain make "betterness" also vagually defined.

# IDEA: 1. annealing of tests in test set
#       2. annealing of constraints

# PROBLEM: 1. Many const semantics or same var semantics in default Koza GP.
#          2. Constraining minimal number of vars and consts.
#          3. More general constraint specification mechanism should be developed with Tree Tries.

# TODO:
#       DONE 1. Testing with caches, probably separate cache enablance.
#       2. Lexicase selection and its advanced forms
#          More advanced form of lexicase that considers pair of axes of interaction CS space
#       3. Unification with discrete domains? Can this work with discrete domains?
#       4. Unification with other evo processes in cde-search: NSGA and coevolution.
#       DONE 5. Tuning of constants
#       6. Syntactic simplifications with axioms (again, need Tree Tries to match rules)
#          No need in tree tries. Rewrites could be done with unification and then replacement.
#          Do we really need syntactic simplification???
#       7. Towards abstract forms (x * x + c * x + c)
#           Reduced to another mutation operator as it replaces any child term with linear combination.
#           Abstract form is just selection of isinstance(term, (Value, Variable)). Generally any child term could be replaced.
#       8. Towards semantic GP (add operators) + propose tuned point operator, using indices
#       9. Math properties and dynamic constraint sets.


#       [BAD IDEA] 10. Gen math expr instead of lisp expr

#      11. Other metrics??? Add when caches are enabled - syntactic diversity (is there convergance to same syntax)
#      13. Aging???
#      14. Dropout ???
#      15. Distribution control in gen_term based on statistics of past decisions at point of generation -
#           First, We need to have metric to see how gen process produce unique terms, not previously found in cache - should be controlled on term build.
#      DONE 16. We observe that classic crossover frequently fallbacks to reproduction - less point-pairs that required num_children in breed.
#          Therefore, we should noto require more generations from pair than present number of crossover points.
#          Better to attempt next parents when budget of crossover is not exhausted.
#          Crossover cache hits --> does it make sense to produce same children? On cache hit - should be no child. Or crossover point should be prohibited.
#          Aging controls which points could crossover.
#          Globally crossover should not work with parents, but only with repository of crossover points (root, term, occur)

#      17. Annealing on present ops - max_counts. At what point to add new op? Based on frequency of cache hits or max gens?
#          Which ops should be first? Should we use fft here?

#      Reorganiziations:
#      DONE 18. Crossover fix --> do not do useless reproductions --> breed reorganization.
#      CURRENT 19. Optim of points as separate mutation operator that can evaluate
#      DONE 20. Inner semantics collection and filtering - rethink of current eval.

# NOTE [BAD IDEA]: we should probably go with const identities: 10 constant - so we allocate 10 identities but have different their bindings ??
# PROBLEM: 1. const identity should have max of 1 presence in the term, it seems that it should be this way, or small number???
#          2. On crossover of const identities, bindings should be transfered to children - should or not??? should
# NOTE: this is bad idea (attempted) - no benefit to move from consts list to dict[Term, dict[int, Tensor]] for consts
#           - it complicates logic and requires additional mandatory binding step.
#       in current implementation we can collect term consts with one traversal if necessary.


# TODO:
#      Crossover without fails with iterating poss
#      ConstOptimization with multidim tensor of consts in one go
#      Solection operators
#      Mutation that is guided by distribution of syntaxes in population ???

# TODO: think about terms that are optimized to consts --> invalid_terms vs const_terms store
# TODO: unification of terms without meta variables to find most abstract common pattern???

# TODO: Debug fail term gen when Finite is disabled, Dedupl is disabled and Const Optim num_evals = 7,

# gen_term should pick op_id based on arity and estimated number of child terms --> create this estimation in Builders, UpToDepth automatic depth calc

if __name__ == "__main__":

    import json

    from operators import CO, PO, RPM, RPX, TS, Dedupl, Elitism, Up2D

    from .torch_alg import alg_ops, koza_1, test_0

    device = "cuda"
    dtype = torch.float16
    rnd_seed = 42

    solver = GPSolver(
        ops=alg_ops,
        device=device,
        dtype=dtype,
        rnd_seed=rnd_seed,
        torch_rnd_seed=rnd_seed,
        max_consts=5,
        max_ops={"inv": 5, "neg": 5},
        with_inner_evals=True,
        # init=RHH(),
        init=Up2D(depth=1),
        # (num_tries=1, lr=0.1)],
        pipeline=[
            # Finite(),
            PO(num_vals=10, frac=1.0, lr=1, syn_simplify=None)
            # syn_simplify=Reduce()),
            # CO(num_vals = 20, lr=1.0,
            #                     num_evals = 7,
            #                     loss_threshold = 1e-2),
            # Elitism(size = 10),
            # TS(),
            # RPM(),
            # RPX(),
            # Dedupl(),
        ],
        # mutations=[PointRandMutation(), PointRandCrossover(), Dedupl(), ConstOptimization1(lr=1.0)],
        # commutative_ops=["add", "mul"],
        forbid_patterns=[
            # "(inv (inv .))",
            # "(neg (neg .))",
            # "(sin (... (sin .)))",
            # "(cos (... (cos .)))",
            # "(exp (... (exp .)))",
            # "(log (... (log .)))",
            # "(... pow (pow . .))",
        ],
        inner_ops_max_counts={
            "sin": {"sin": 0},
            "cos": {"cos": 0},
            "exp": {"exp": 0},
            "log": {"log": 0},
            "pow": {"pow": 1},
            "inv": {"inv": 1},
        },
        immediate_arg_limits={
            "inv": {"inv": 0},
            "neg": {"neg": 0},
            "log": {"exp": 0},
            "exp": {"log": 0},
        },
    )

    free_vars, target = test_0.sample_set("train", device=device, dtype=dtype, generator=solver.torch_gen, sorted=True)

    solver.fit(free_vars, target)

    with open("gp-metrics.json", "w") as f:
        json.dump(solver.metrics, indent=4, default=str, fp=f)

    # print("Metrics:\n", metrics_json)

    pass
