''' Syntax cache with builders '''


import inspect
from typing import Callable, Optional, Type

import numpy as np
import torch

from t_search.base import ServiceBase
from t_search.syntax.generation import Builder, Builders, TermGenContext, gen_all_terms, get_fn_arity, grow
from t_search.syntax.replacement import replace_fn, replace_path, replace_pos, replace_pos_protected
from t_search.syntax.stats import get_depth, get_positions
from t_search.syntax.str import parse_const_skeleton_to_term, parse_term
from t_search.syntax.term import Op, Term, TermPos, Value, Variable
from t_search.syntax.traverse import postorder_map
from t_search.syntax.unification import UnifyBindings, match_root
from t_search.syntax.validation import get_counts, get_pos_constraints, get_pos_sibling_counts, is_valid
from t_search.utils import GLOBAL_RNG


class Syntax(ServiceBase):

    def __init__(self, *,
                 # from solver context
                 var_names: list[str],
                 ops_signatures: dict[str, inspect.Signature],
                 ops_schedule: dict[str, int], # on what iteration the operation should be activated (used in random syntax generation)
                 get_cur_gen: Callable[[], int],
                 add_metrics: Callable,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.bfloat16,
                 
                 # from config params
                 max_term_depth=17,
                 min_consts: int = 0,
                 max_consts: int = 10,  # 0 to disable consts in terms
                 min_vars: int = 1,
                 max_vars: int = 10,  # max number of free variables
                 const_range: torch.Tensor,
                 forbid_patterns: list[str] = [],
                 ops_counts: dict[str, tuple[int, int]] = {},
                 inner_ops_max_counts: dict[str, dict[str, int]] = {},
                 immediate_arg_limits: dict[str, dict[str, int]] = {},
                 prohibit_ops_on_consts_only: bool = True,       
                 capacity: int = 1024,
                 rnd: np.random.Generator = GLOBAL_RNG,
                 torch_gen: torch.Generator | None = None
                 ):
        self.ops_signatures = ops_signatures
        self.add_metrics = add_metrics

        self.max_term_depth = max_term_depth    
        self.min_consts = min_consts
        self.max_consts = max_consts
        self.min_vars = min_vars
        self.max_vars = max_vars

        self.device = device
        self.dtype = dtype
        self.rnd = rnd
        self.torch_gen = torch_gen
        self.capacity = capacity
        self.waiting_ops: set[str] = set(ops_schedule.keys())
        self.ops_schedule = sorted([(v, k) for k, v in ops_schedule.items()])
        self.get_cur_gen = get_cur_gen

        self.const_range = const_range

        self.ops_counts = ops_counts
        self.prohibit_ops_on_consts_only = prohibit_ops_on_consts_only
        self.inner_ops_max_counts = inner_ops_max_counts
        self.immediate_arg_limits = immediate_arg_limits

        self.forbid_patterns = forbid_patterns
        self.match_cache: dict[tuple, UnifyBindings] = {}
        self.fpatterns = []
        if len(self.forbid_patterns) > 0:
            for p in self.forbid_patterns:
                t, i = parse_term(p)
                assert len(p) == i, f"Invalid pattern: {p}"
                self.fpatterns.append(t)


        builders: dict[Type | str, Builder] = {}
        self.const_builder = Builder("C", self._alloc_const, 0, self.min_consts, self.max_consts)
        builders[Value] = self.const_builder
        self.var_builder = Builder("x", self._alloc_var, 0, self.min_vars, self.max_vars)
        builders[Variable] = self.var_builder

        self.op_builders: dict[str, Builder] = {}
        for op_id, op_signatrue in self.ops_signatures.items():
            op_arity = get_fn_arity(op_signatrue)
            op_builder = Builder(
                op_id,
                self._alloc_op_builder(op_id),
                op_arity
            )
            # commutative = op_id in self.commutative_ops)
            if op_id in self.ops_counts:
                op_min_count, op_max_count = self.ops_counts[op_id]
                op_builder.min_count = op_min_count
                op_builder.max_count = op_max_count

            self.op_builders[op_id] = op_builder        

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

        self.pos_cache: dict[Term, list[TermPos]] = {}
        self.pos_context_cache: dict[
            Term,
            dict[tuple[Term, int], TermGenContext],
        ] = {}
        self.depth_cache: dict[Term, int] = {}
        self.counts_cache: dict[Term, np.ndarray] = {}

        self.const_id: int = 0
        self.const_tape: torch.Tensor = torch.empty(0, device=self.device, dtype=self.dtype)
        self.syntax: dict[tuple[str, Term], Term] = {}
        self.vars = [Variable(var_id) for var_id in var_names]
        self.vars_dict = {x.var_id: x for x in self.vars}

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

        self.builders.disable_builders(list(self.waiting_ops)) 
        
        self.zero_value = self.const_builder.fn(value=0.0)
        self.skeletons_cache: dict[Term, Term] = {}

    def _alloc_const(self, *, value: Optional[float | torch.Tensor] = None) -> Value:
        if self.const_id >= self.const_tape.shape[0]:
            delta = max(1, self.max_consts) * self.capacity
            new_tape = torch.empty(
                self.const_tape.shape[0] + delta,
                device=self.device,
                dtype=self.dtype)
            new_tape[: self.const_tape.shape[0]] = self.const_tape
            new_rands = torch.rand(
                delta,
                device=self.device,
                dtype=self.dtype,
                generator=self.torch_gen,
            )
            if self.const_range is not None:
                dist = self.const_range[1] - self.const_range[0]
                new_rands *= dist
                new_rands += self.const_range[0]
            new_tape[self.const_tape.shape[0] :] = new_rands
            self.const_tape = new_tape
            del new_rands
        if value is not None:  # const value provided - no alloc of consts
            if value < -1e30:
                value = -torch.inf
            if value > 1e30:
                value = torch.inf
            self.const_tape[self.const_id] = value
            # if not torch.is_tensor(value):
            #     value = torch.tensor(value, dtype=self.dtype, device=self.device)
            # return Value(value)
        value = self.const_tape[self.const_id]
        self.const_id += 1
        return Value(value)

    def _alloc_var(self, *, var_id: Optional[str] = None) -> Variable:
        if var_id is not None:
            var = self.vars_dict.get(var_id, None)
            if var is not None:
                return var
        var = self.rnd.choice(self.vars)
        return var
    
    def _validate_patterns(self, term: Term) -> bool:
        for fpattern in self.fpatterns:
            match = match_root(term, fpattern, prev_matches=self.match_cache)
            if match is not None:
                return False
        return True    
    
    def _alloc_op_builder(self, op_id: str) -> Callable:

        def _alloc_op(*args):
            signature = (op_id, *args)
            if signature in self.syntax:
                self.add_metrics(syntax_hit=1)
                term = self.syntax[signature]
            else:
                self.add_metrics(syntax_miss=1)
                term = Op(op_id, args)
                self.syntax[signature] = term
            if not self._validate_patterns(term):
                self.syntax.pop(signature, None)
                return None
            # if self.invalid_terms.is_invalid(term):
            #     # NOTE that we increase on every cache hit
            #     self.add_metrics(invalid_hit=1)
            #     return None  # do not output known invalid terms
            # # elif term in self.const_term_outputs:
            # #     self.metrics["syntax_const"] = self.metrics.get("syntax_const", 0) + 1
            # #     # return Value(self.const_term_outputs[term]) # return const value
            # #     # return None
            # #     pass
            # #     # NOTE: returning value could ruin constraints. Instead, we disallow constant terms because constant leaf could be used instead.
            # #     # TODO: separate operator that transforms terms to simple forms with removed constants
            return term

        # else:
        #     def _alloc_op(*args):
        #         self.metrics[miss_key] = self.metrics.get(miss_key, 0) + 1
        #         term = Op(op_id, args)
        #         if self.validate_term(term):
        #             return term
        #         return None

        return _alloc_op
    
    def parse_term_str(self, term_str: str) -> Term | None:
        try:
            term, _ = parse_term(term_str)
            def map_term(t: Term, args: list[Term]) -> Term:
                if isinstance(t, Variable):
                    return self.var_builder.fn(var_id=t.var_id)
                if isinstance(t, Value):
                    return self.const_builder.fn(value=t.value)
                builder = self.builders.get_term_builder(t)                    
                return builder.fn(*args)
            cached_term = postorder_map(term, map_term)
            return cached_term
        except Exception:
            return None 
        
    def parse_const_skeleton(self, skeleton_str: str, const_name: str = "C") -> Term | None:
        try:
            term, _ = parse_const_skeleton_to_term(skeleton_str, const_name=const_name)
            def map_term(t: Term, args: list[Term]) -> Term:
                if isinstance(t, Variable):
                    return self.var_builder.fn(var_id=t.var_id)
                if isinstance(t, Value):
                    return self.const_builder.fn(value=t.value)                
                builder = self.builders.get_term_builder(t)                    
                return builder.fn(*args)
            cached_term = postorder_map(term, map_term)
            return cached_term
        except Exception:
            return None        
        
    def get_skeleton(self, term: Term) -> Term:
        ''' Replaces all constants to specified const_value forming unified skeleton for family of terms'''
        if term in self.skeletons_cache:
            return self.skeletons_cache[term]
        def replace_consts(t: Term, *_):
            if isinstance(t, Value):
                return self.zero_value
            return None
        skeleton = self.replace_fn(term, replace_consts)
        self.skeletons_cache[term] = skeleton
        return skeleton

    def get_depth(self, term: Term) -> int:
        term_depth = get_depth(term, self.depth_cache)
        return term_depth
    
    def get_counts(self, term: Term):
        counts = get_counts(term, self.builders, self.counts_cache)
        return counts
    
    def get_num_consts(self, term: Term) -> int:
        counts = self.get_counts(term)
        num_consts = counts[self.const_builder.id].item()
        return num_consts
    
    def get_size(self, term: Term):
        counts = self.get_counts(term)
        size = counts.sum().item()
        return size

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
    
    # def replace_positions_unvalidated(self, term: Term, pos: TermPos, new_subterms: list[Term]) -> list[Term]:
    #     ''' Replace same position with multiple new subterms, without validation '''
    #     children = replace_pos(pos, new_subterm, self.builders)
    #     return children
    
    def replace_path_unvalidated(self, term: Term, pos: TermPos, new_subterms: list[Term]) -> list[Term]:
        if pos is None or len(new_subterms) == 0:
            return []
        return replace_path(pos, new_subterms, self.builders)
    
    def replace_fn(self, term: Term, get_replacement_fn: Callable[[Term, int], Optional[Term]]) -> Term:
        replaced_term = replace_fn(term, get_replacement_fn, self.builders)
        return replaced_term

    def is_valid(self, term: Term) -> bool:
        ''' Checks validity according to syntactic constraints only (not evaluations)'''
        term_depth = self.get_depth(term)
        if term_depth > self.max_term_depth:
            return False
        # eval_valid = not self.invalid_terms.is_invalid(term)
        # if not eval_valid:
        #     return False
        term_is_valid = is_valid(term, builders=self.builders, counts_cache=self.counts_cache)
        if not term_is_valid:
            return False
        pattern_valid = self._validate_patterns(term)
        return pattern_valid
    
    def get_const(self, value) -> Value: 
        const_term = self.const_builder.fn(value=value)
        return const_term
    
    def get_var(self, var_id: str) -> Variable:
        var_term = self.var_builder.fn(var_id=var_id)
        return var_term
    
    def has_op(self, op_id: str) -> bool:
        return op_id in self.op_builders
    
    def get_op(self, op_id: str, *args) -> Op | None:
        if any(arg is None for arg in args):
            return None
        arity = self.op_builders[op_id].arity()
        if arity == 1 and len(args) != 1:
            return None
        cur_args = args[:arity]
        left_args = args[arity:]
        cur_term = self.op_builders[op_id].fn(*cur_args)
        while len(left_args) > 0:
            new_args = [cur_term, *left_args[:arity - 1]]
            cur_term = self.op_builders[op_id].fn(*new_args)
            left_args = left_args[arity - 1:]
        # if check_validity and new_term is not None:
        #     if not self.is_valid(new_term):
        #         return None
        return cur_term
    
    def get_rand_op(self, arg_builder: Callable[[int], Term]) -> Op | None:
        op_builder = self.rnd.choice(list(self.op_builders.values()))
        args = [arg_builder(arg_i) for arg_i in range(op_builder.arity())]
        op = op_builder.fn(*args)
        return op
    
    def get_vars(self) -> list[Variable]: 
        return self.vars
    
    def grow(self, max_term_depth: int | None = None,
             start_pos: tuple[Term, TermPos] | None = None,
             grow_leaf_prob: float | None = None,
             freq_skew: bool = False) -> Term:
        cur_iter = self.get_cur_gen()
        ops_update = []
        while len(self.waiting_ops) > 0 and len(self.ops_schedule) > 0:
            sched_iter, sched_op = self.ops_schedule[0]
            if sched_iter <= cur_iter:
                ops_update.append(sched_op)
                self.waiting_ops.remove(sched_op)
                self.ops_schedule.pop(0)
            else:
                break
        if len(ops_update) > 0:
            self.builders.enable_builders(ops_update)
            self.pos_context_cache.clear()
        if start_pos is not None:
            term, position = start_pos
            start_context, arg_counts = self.get_gen_constraints(term, position)
        else:
            start_context = None
            arg_counts = None
        max_term_depth = min(max_term_depth, self.max_term_depth) if max_term_depth is not None else self.max_term_depth
        new_term = grow(builders = self.builders, 
                        grow_depth = max_term_depth,
                        grow_leaf_prob = grow_leaf_prob,
                        rnd = self.rnd,
                        start_context=start_context,
                        arg_counts=arg_counts,
                        add_metrics = self.add_metrics,
                        freq_skew=freq_skew)
        return new_term
    
    def get_all_terms(self, up2depth: int = 2, max_consts: int = 0) -> list[Term]:
        gen_context = TermGenContext(self.builders.default_gen_context.min_counts,
                                        self.builders.default_gen_context.max_counts.copy(),
                                        self.builders.default_gen_context.arg_limits)
        gen_context.max_counts[self.const_builder.id] = max_consts  # limit constants
        terms = gen_all_terms(self.builders, depth=up2depth, start_context=gen_context)
        return terms
                
    def get_finalizer(self):
        self.add_metrics(consts_used=self.const_id)
        def finalizer():
            del self.const_tape
            self.const_id = 0
        return finalizer

    def get_iter_metrics(self):
        iter_num_syntax = len(self.syntax)
        iter_num_consts = self.const_id
        return {
            'iter_num_syntax': [iter_num_syntax],
            'iter_num_consts': [iter_num_consts]
        }