""" GP solver, configurable with different operators
"""

from time import perf_counter
from typing import Any, Callable, Literal, Optional, Sequence, Type

import numpy as np
import torch

from .utils import GLOBAL_RNG, EvSearchTermination, GPSolverStatus, add_metric, stack_rows, timed
from .operators import RHH, RPM, RPX, TS, Initialization, Operator, Listener
from sklearn.base import BaseEstimator, RegressorMixin

from t_search.syntax.traverse import postorder_map

# from mutation import CO, Dedupl, Mutation, PO, RPX, RPM, Reduce
from t_search.syntax import Op, Term, TermPos, Value, Variable, parse_term, parse_const_skeleton_to_term
from t_search.syntax.generation import Builder, Builders, TermGenContext, get_fn_arity
from t_search.syntax.unification import UnifyBindings, match_root
from t_search.syntax.stats import get_depth, get_positions
from t_search.syntax.validation import get_counts, get_pos_constraints, get_pos_sibling_counts, is_valid
from t_search.syntax.replacement import replace_pos, replace_pos_protected
from t_search.evaluators import Evaluator, Evaluations


# operational semantics of default symbols
default_alg_ops = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "pow": lambda a, b: a ** b,
    "neg": lambda a: -a,
    "inv": lambda a: 1 / a,
    "exp": lambda a: torch.exp(a),
    "log": lambda a: torch.log(a),
    "sin": lambda a: torch.sin(a),
    "cos": lambda a: torch.cos(a),
}

class GPSolver(BaseEstimator, RegressorMixin):

    def __init__(
        self, *,
        ops: dict[str, Callable] | list[str] = default_alg_ops,
        evaluator: Evaluator,
        init: Initialization = RHH(),
        operators: list["Operator"] = [TS(), RPM(), RPX()],
        ops_counts: dict[str, tuple[int, int]] = {},
        forbid_patterns: list[str] = [],
        inner_ops_max_counts: dict[str, dict[str, int]] = {},
        immediate_arg_limits: dict[str, dict[str, int]] = {},
        prohibit_ops_on_consts_only: bool = True,
        max_term_depth=17,
        min_consts: int = 0,
        max_consts: int = 10,  # 0 to disable consts in terms
        min_vars: int = 1,
        max_vars: int = 10,  # max number of free variables
        max_gen: int = 100,
        pop_size: int = 1000,
        listeners: list[Listener] = [],
        const_range: Optional[tuple[float, float]] = None,  # if not set, computed from X, y
        rnd: np.random.Generator = GLOBAL_RNG,
        torch_gen: Optional[torch.Generator] = None,
        device:Literal["cpu", "cuda"]="cpu",
        dtype:torch.dtype=torch.float32,
        debug: bool = False
    ):
        self.debug = debug

        self.metrics = {}

        if type(ops) is list:
            op_dict = {}            
            for op_id in ops:
                if op_id in default_alg_ops:
                    op_dict[op_id] = default_alg_ops[op_id]
                else:
                    raise ValueError(f"Operator {op_id} is not in default set. Supported: {list(default_alg_ops.keys())}")
            ops = op_dict
        for op_id, op_fn in ops.items():
            if op_fn is None: # use default impl 
                if op_id in default_alg_ops:
                    ops[op_id] = default_alg_ops[op_id]
                else:
                    raise ValueError(f"Operator {op_id} is not in default set. Supported: {list(default_alg_ops.keys())}")
                
        self.ops = ops

        self.evaluator = evaluator
        self.ops_counts = ops_counts

        self.max_term_depth = max_term_depth
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.min_consts = min_consts
        self.max_consts = max_consts
        self.forbid_patterns = forbid_patterns
        self.match_cache: dict[tuple, UnifyBindings] = {}
        self.fpatterns = []
        if len(self.forbid_patterns) > 0:
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

        self.init = init
        self.operators = operators
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.device = device
        self.dtype = dtype
        self.prohibit_ops_on_consts_only = prohibit_ops_on_consts_only
        self.inner_ops_max_counts = inner_ops_max_counts
        self.immediate_arg_limits = immediate_arg_limits

        self.rnd = rnd 
        self.torch_gen = torch_gen

        # next are runtime fields and caches that works across fit calls
        self.target: torch.Tensor = torch.empty(
            0,
            device=self.device,
            dtype=self.dtype,
        )

        self.new_listener_terms: list[Term] = []
        self.pos_cache: dict[Term, list[TermPos]] = {}
        self.pos_context_cache: dict[
            Term,
            dict[tuple[Term, int], TermGenContext],
        ] = {}
        self.depth_cache: dict[Term, int] = {}
        self.counts_cache: dict[Term, np.ndarray] = {}

        self.gen: int = 0
        self.is_fitted_: bool = False
        self.status: GPSolverStatus = "INIT"
        self.start_time: float = 0
        self.const_id: int = 0
        self.const_tape: torch.Tensor = torch.empty(0, device=self.device, dtype=self.dtype)
        self.listeners: list[Listener] = listeners
        self.syntax: dict[tuple[str, Term], Term] = {}

        self.vars: dict[str, Variable] = {}
        self.var_binding: dict[str, torch.Tensor] = {}

        self.op_builders: dict[str, Builder] = {}
        for op_id, op_fn in self.ops.items():
            op_arity = get_fn_arity(op_fn)
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

    def on_start(
        self,
        free_vars: Sequence | torch.Tensor,
        target: Sequence | torch.Tensor,
    ):
        """Called before each fit"""

        self.syntax.clear()
        self.pos_cache.clear()
        self.pos_context_cache.clear()
        self.counts_cache.clear()
        self.depth_cache.clear()

        self.gen = 0
        self.metrics = {}
        self.status = "INIT"
        self.start_time = perf_counter()
        self.is_fitted_ = False
        builders: dict[Type | str, Builder] = {}

        self.const_id = 0
        del self.const_tape
        self.const_tape = torch.empty(0, device=self.device, dtype=self.dtype)

        # self.const_builder = None
        # if self.max_consts > 0:
        self.const_builder = Builder("C", self._alloc_const, 0, self.min_consts, self.max_consts)
        builders[Value] = self.const_builder

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

        self.evaluator.on_start(self)

        if self.const_range is None:
            self.const_range = self.evaluator.detect_const_range(self.target, self.var_binding.values())

        self.new_listener_terms.clear()

        for listener in self.listeners:
            listener.on_start(self)

        for op in self.operators:
            op.on_start(self)

        pass

    def _get_vars(self, free_vars):
        vars: list[Variable] = []
        var_binding: dict[str, torch.Tensor] = {}
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
        if self.const_id >= self.const_tape.shape[0]:
            delta = max(1, self.max_consts) * self.pop_size
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
            self.const_tape[self.const_id] = value
            # if not torch.is_tensor(value):
            #     value = torch.tensor(value, dtype=self.dtype, device=self.device)
            # return Value(value)
        value = self.const_tape[self.const_id]
        self.const_id += 1
        return Value(value)

    def parse_term_str(self, term_str: str) -> Term | None:
        try:
            term, _ = parse_term(term_str)
            def map_term(t: Term, args: list[Term]) -> Term:
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
                builder = self.builders.get_term_builder(t)                    
                return builder.fn(*args)
            cached_term = postorder_map(term, map_term)
            return cached_term
        except Exception:
            return None        
    
    def add_metric(self, *, scope: str = "", **kwargs):
        cur_metrics = self.metrics if scope == "" else self.metrics.setdefault(scope, {})
        add_metric(cur_metrics, **kwargs)

    def _breed(self, population: list[Term]) -> list[Term]:
        """Pipeline that mutates parents and then applies crossover on pairs. One-point operations"""

        children = population

        for operator in self.operators:
            
            # validation - disable in production for speed
            if self.debug:
                for t in children:
                    assert self.is_valid(t), f"Invalid term before operator {operator.name}: {t}"

            children, elapsed = timed(operator)(self, children)
            self.add_metric(scope=operator.name, step_time=[elapsed], total_time=elapsed)

        return children

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
                self.add_metric(syntax_hit=1)
                term = self.syntax[signature]
            else:
                self.add_metric(syntax_miss=1)
                term = Op(op_id, args)
                self.syntax[signature] = term
            if not self._validate_patterns(term):
                self.syntax.pop(signature, None)
                return None
            if self.evaluator.is_invalid(term):
                # NOTE that we increase on every cache hit
                self.add_metric(invalid_hit=1)
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
    
    def _loop(self):
        population, elapsed = timed(self.init)(self, self.pop_size)
        self.add_metric(init_time=elapsed)
        _, elapsed = timed(self.evaluator.eval)(population)
        self.add_metric(init_eval_time=elapsed, total_eval_time=elapsed)

        while self.gen < self.max_gen:
            for listener in self.listeners:
                listener.on_gen_start(self, self.gen, population)            
            population = self._breed(population)
            _, elapsed = timed(self.evaluator.eval)(population)
            self.add_metric(gen_eval_time=[elapsed], total_eval_time=elapsed)
            for listener in self.listeners:
                listener.on_gen_end(self, self.gen, population)
            self.gen += 1

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
        eval_valid = not self.evaluator.is_invalid(term)
        return term_is_valid and pattern_valid and eval_valid

    def _check_trivial(self, raise_on_solution: bool = False) -> bool:
        ''' If target is constant or var, evaluator will rise 'Solution Found' error '''
        try:
            const_val = self.evaluator.is_const(self.target)
            if const_val is not None:  # NOTE: or torch.any ??? config option
                const_term = self.const_builder.fn(value=const_val)  # len(self.const_binding))
                self.evaluator.eval(const_term)
            self.evaluator.eval(list(self.vars.values()))
        except EvSearchTermination as e:
            if e.status == "SOLVED" and not raise_on_solution:
                return True
            raise e
        return False
    
    def eval(
        self,
        terms: Sequence[Term] | Term,
        *,
        return_outputs: Literal["list", "tensor"] = "list",
        return_fitness: Literal["none", "list", "tensor"] = "none",
    ) -> Evaluations:
        evaluations = self.evaluator.eval(
            terms, self,
            return_outputs=return_outputs,
            return_fitness=return_fitness,
        )
        return evaluations

    def get_counts(self, term: Term):
        counts = get_counts(term, self.builders, self.counts_cache)
        return counts
    
    def get_size(self, term: Term):
        counts = self.get_counts(term)
        size = counts.sum().item()
        return size
    
    def on_end(self):
        
        self.add_metric(
            gen = self.gen,
            final_time = round((perf_counter() - self.start_time) * 1000),
            status = self.status,
            consts = self.const_id,
        )

        for op in self.operators:
            op.on_end(self)

        for listener in self.listeners:
            listener.on_end(self)

        self.evaluator.on_end(self)

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> "GPSolver":
        """
        Fit the solver to the data.

        Args:
            X (array-like): Input features.
            y (array-like): Target labels.

        Returns:
            self: Returns the instance itself.
        """
        self.on_start(free_vars=X, target=y)
        try:
            self._check_trivial(raise_on_solution=True)
            self._loop()
        except EvSearchTermination as e:
            self.status = e.status
        self.is_fitted_ = True
        if self.status == "INIT":
            self.status = "MAX_GEN"

        self.on_end()

        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:

        if not self.is_fitted_:
            raise RuntimeError("Solver is not fitted yet")

        _, var_binding = self._get_vars(X)

        output: torch.Tensor = self.evaluator.predict(var_binding, ops = self.ops)
        if output is None:
            raise RuntimeError("Evaluation of the best term returned None, not all terminals may be bound")
        output_numpy = output.cpu().numpy()
        return output_numpy
    
    def save_metrics(self, filepath: str):
        """Saves metrics to a JSON file."""
        import json

        def metrics_serializer(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            if isinstance(obj, Term):
                return str(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=4, default=metrics_serializer)


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