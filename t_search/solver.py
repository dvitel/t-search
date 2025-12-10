""" GP solver, configurable with different operators
"""

from functools import partial
import inspect
from time import perf_counter
from typing import Any, Callable, Literal, Sequence, Type

import numpy as np
import torch

from t_search.base import ServiceBase
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.semantics import Semantics
from t_search.operators.initialization import Initialization
from t_search.operators.listeners import GenListener
from t_search.solver_utils import get_method_params, read_config, register_services
from t_search.syntax.syntax import Syntax

from .utils import GLOBAL_RNG, EvSearchTermination, GPSolverStatus, add_metrics, timed
from .operators import Operator
from sklearn.base import BaseEstimator, RegressorMixin


# from mutation import CO, Dedupl, Mutation, PO, RPX, RPM, Reduce
from t_search.syntax import Term
from t_search.evaluators import Evaluator

def detect_const_range(target: torch.Tensor, var_bindings: Sequence[torch.Tensor]) -> torch.Tensor:
    min_value = target.min()
    max_value = target.max()
    if torch.isclose(
        min_value,
        max_value,
    ):
        min_value = min_value - 0.1
        max_value = max_value + 0.1
    const_range = torch.tensor([min_value, max_value], dtype=target.dtype, device=target.device)
    free_vars_as_one = torch.stack(tuple(var_bindings), dim=0)
    min_fv = torch.min(free_vars_as_one)
    max_fv = torch.max(free_vars_as_one)
    const_range[0] = torch.minimum(const_range[0], min_fv)
    const_range[1] = torch.maximum(const_range[1], max_fv)
    dist = const_range[0] - const_range[1]
    const_range[0] -= 0.1 * dist
    const_range[1] += 0.1 * dist
    return const_range

def get_var_bindings(free_vars: torch.Tensor) -> dict[str, torch.Tensor]:
    var_bindings: dict[str, torch.Tensor] = {}
    for i, fv in enumerate(free_vars):
        var_id = f"x{i}"
        var_bindings[var_id] = fv
    return var_bindings

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

    def __init__(self, *,
        service_definitions: dict[list, dict],
        init_service_name: str,
        operator_service_names: list[str],
        evaluator_service_name: str,
        syntax_service_name: str,
        fitness_service_name: str,
        semantics_service_name: str,

        ops: dict[str, Callable] | list[str] = default_alg_ops,

        max_gen: int = 100,

        device:Literal["cpu", "cuda"]="cpu",
        dtype:torch.dtype=torch.float32,
        rnd: np.random.Generator = GLOBAL_RNG,
        torch_gen: torch.Generator | None = None,
        debug: bool = False,
    ):
        
        self.service_definitions = service_definitions
        self.init_service_name = init_service_name
        self.operator_service_names = operator_service_names
        self.evaluator_service_name = evaluator_service_name
        self.syntax_service_name = syntax_service_name
        self.fitness_service_name = fitness_service_name
        self.semantics_service_name = semantics_service_name

        self.services = {}

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
        self.ops_signatures = {op_id: inspect.signature(op_fn) for op_id, op_fn in ops.items()}

        self.max_gen = max_gen
        self.device: torch.device = torch.device(device)
        self.dtype: torch.dtype = dtype

        self.rnd: np.random.Generator = rnd
        self.torch_gen: torch.Generator = torch_gen if torch_gen is not None else torch.Generator(device = device)

        self.free_vars: torch.Tensor | None = None
        self.target: torch.Tensor | None = None 
        self.const_range: torch.Tensor | None = None

        self.gen: int = 0
        self.is_fitted_: bool = False
        self.status: GPSolverStatus = "INIT"
        self.start_time: float = 0

        self.init: Initialization | None = None 
        self.operators: list[Operator] = []
        self.evaluator: Evaluator | None = None
        self.syntax: Syntax | None = None
        self.fitness: Fitness | None = None
        self.semantics: Semantics | None = None
        self.gen_listeners: list[GenListener] = []
        self.cur_population: list[Term] = []

    def on_start(
        self,
        free_vars: Sequence | torch.Tensor,
        target: Sequence | torch.Tensor,
    ):
        """ Creating all services from definitions andd instantiate the pipeline.
            Called once on fit. If some checkpoint is utilized - should be in service configs.
        """

        self.is_fitted_ = False
        self.status = "INIT"
        self.gen = 0
        self.metrics = {}
        self.start_time = perf_counter()

        if not torch.is_tensor(free_vars):
            free_vars = torch.tensor(
                free_vars,
                device=self.device,
                dtype=self.dtype,
            )

        self.free_vars = free_vars.to(device=self.device, dtype=self.dtype)

        self.var_bindings = get_var_bindings(self.free_vars)
        self.var_names = list(self.var_bindings.keys())

        if not torch.is_tensor(target):
            target = torch.tensor(
                target,
                device=self.device,
                dtype=self.dtype,
            )

        self.target = target.to(device=self.device, dtype=self.dtype)

        self.const_range = detect_const_range(self.target, self.free_vars)

        default_context = {
            "rnd": self.rnd,
            "torch_gen": self.torch_gen,
            "device": self.device,
            "dtype": self.dtype,
            "ops": self.ops,
            "ops_signatures": self.ops_signatures,
            "free_vars": self.free_vars,
            "target": self.target,
            "dims": self.target.shape[0],
            "const_range": self.const_range,
            "term_order": lambda term: self.syntax.get_size(term),
            "max_gen": self.max_gen,
            "get_cur_gen": lambda: self.gen,
            "get_cur_population": lambda: self.cur_population,
            "var_bindings": self.var_bindings,
            "var_names": self.var_names,
            "debug": self.debug,
        }

        self.services.clear()

        def service_builder(service_name: str, service_cls: Type, params: dict) -> Any:

            service_params = get_method_params(service_cls, "__init__")

            service_context = {}            

            for param_name in service_params:
                # if param_name in self.services:
                #     service_context[param_name] = self.services[param_name]
                if param_name in default_context:
                    service_context[param_name] = default_context[param_name]
                elif param_name == "add_metrics":
                    service_context[param_name] = partial(self.add_metrics, scope=service_name)
                elif param_name in params:
                    service_context[param_name] = params[param_name]
                elif param_name in ("fitness", "semantics", "syntax", "evaluator") and param_name in self.services:
                    service_context[param_name] = self.services[param_name]

            inited_params = set(service_context.keys())
            left_params = []
            default_params = {}
            for p, (has_default, default_opt) in service_params.items():
                if p not in inited_params:
                    if has_default:
                        default_params[p] = default_opt
                    else:
                        left_params.append(p)
            if len(left_params) > 0:
                raise ValueError(f"Cannot build service '{service_name}': missing parameters {left_params}")
                
            if self.debug:
                print(f"Building {service_name}:{service_cls.__name__}:")
                for k, v in service_context.items():
                    if k == "add_metrics":
                        print(f"\t{k}: <function>")
                    else:
                        print(f"\t{k}: {v}")
                if len(default_params) > 0:
                    print(f"---- Default params:")
                    for k, v in default_params.items():
                        print(f"\t{k}: {v}")
                print(f"--------------------------------")
            service = service_cls(**service_context)
            return service

        register_services(self.service_definitions, self.services, service_builder)

        if self.init_service_name not in self.services:
            raise ValueError(f"Init service '{self.init_service_name}' not found among registered services")

        self.init = self.services[self.init_service_name]
        if not isinstance(self.init, Initialization):
            raise ValueError(f"Init service '{self.init_service_name}' is not Initialization instance")
                
        self.operators = []
        for op_name in self.operator_service_names:
            if op_name not in self.services:
                raise ValueError(f"Operator service '{op_name}' not found among registered services")
            operator = self.services[op_name]
            if not isinstance(operator, Operator):
                raise ValueError(f"Operator service '{op_name}' is not Operator instance")
            self.operators.append(operator)

        if self.evaluator_service_name not in self.services:
            raise ValueError(f"Evaluator service '{self.evaluator_service_name}' not found among registered services")
        
        self.evaluator = self.services[self.evaluator_service_name]
        if not isinstance(self.evaluator, Evaluator):
            raise ValueError(f"Evaluator service '{self.evaluator_service_name}' is not Evaluator instance")
        
        if self.syntax_service_name not in self.services:
            raise ValueError(f"Syntax service '{self.syntax_service_name}' not found among registered services")
        
        self.syntax = self.services[self.syntax_service_name]
        if not isinstance(self.syntax, Syntax):
            raise ValueError(f"Syntax service '{self.syntax_service_name}' is not Syntax instance")
        
        if self.fitness_service_name not in self.services:
            raise ValueError(f"Fitness service '{self.fitness_service_name}' not found among registered services")
        self.fitness = self.services[self.fitness_service_name]
        if not isinstance(self.fitness, Fitness):
            raise ValueError(f"Fitness service '{self.fitness_service_name}' is not Fitness instance")
        
        if self.semantics_service_name not in self.services:
            raise ValueError(f"Semantics service '{self.semantics_service_name}' not found among registered services")
        self.semantics = self.services[self.semantics_service_name]
        if not isinstance(self.semantics, Semantics):
            raise ValueError(f"Semantics service '{self.semantics_service_name}' is not Semantics instance")

        for service in self.services.values():
            if isinstance(service, ServiceBase):
                service.init()

        self.gen_listeners = []
        for service in self.services.values():
            if isinstance(service, GenListener):
                self.gen_listeners.append(service)

        pass
    
    def add_metrics(self, *, scope: str = "", **kwargs):
        cur_metrics = self.metrics if scope == "" else self.metrics.setdefault(scope, {})
        add_metrics(cur_metrics, **kwargs)
    
    def _loop(self):
        initial_population, elapsed = timed(self.init)()
        self.cur_population = initial_population
        self.add_metrics(init_time=elapsed)
        _, elapsed = timed(self.evaluator.eval)(self.cur_population)
        self.add_metrics(init_eval_time=elapsed, total_eval_time=elapsed)

        while self.gen < self.max_gen:
            iter_start_time = perf_counter()
            for listener in self.gen_listeners:
                listener.on_gen_start(self.gen, self.cur_population)            

            children = self.cur_population

            for operator_name, operator in zip(self.operator_service_names, self.operators):
                
                # validation - disable in production for speed
                if self.debug:
                    for t in children:
                        assert self.syntax.is_valid(t), f"Invalid term before operator {operator_name}: {t}"

                children, elapsed = timed(operator)(self, children)
                self.add_metrics(scope=operator_name, step_time=[elapsed], total_time=elapsed)

            self.cur_population = children
                
            # _, elapsed = timed(self.evaluator.eval)(self.cur_population)
            # self.add_metrics(gen_eval_time=[elapsed], total_eval_time=elapsed)
            for listener in self.gen_listeners:
                listener.on_gen_end(self.gen, self.cur_population)
            iter_end_time = perf_counter()
            self.add_metrics(iter_time=[round((iter_end_time - iter_start_time) * 1000)])
            self.gen += 1

    def _check_trivial(self, raise_on_solution: bool = False) -> bool:
        ''' If target is constant or var, evaluator will rise 'Solution Found' error '''
        try:
            const_val = self.evaluator.is_const(self.target)
            if const_val is not None:
                const_term = self.syntax.get_const(value=const_val)
                self.evaluator.eval(const_term)
            self.evaluator.eval(self.syntax.get_vars())
        except EvSearchTermination as e:
            if e.status == "SOLVED" and not raise_on_solution:
                return True
            raise e
        return False
    
    def on_end(self):
        
        self.add_metrics(
            gen = self.gen,
            final_time = round((perf_counter() - self.start_time) * 1000),
            status = self.status,
            # consts = self.const_id,
        )

        finalizers = []
        for service in self.services.values():
            if isinstance(service, ServiceBase):
                finalizer = service.get_finalizer()
                if finalizer is not None:
                    finalizers.append(finalizer)

        for finalizer in finalizers:
            finalizer()

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

        if not torch.is_tensor(X):
            X = torch.tensor(
                X,
                device=self.device,
                dtype=self.dtype,
            )

        free_vars = X.to(device=self.device, dtype=self.dtype)

        var_bindings = get_var_bindings(free_vars)

        output: torch.Tensor = self.evaluator.test(var_bindings, ops = self.ops)
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

def config_pipeline(*, dataset:str, config: str, output="koza-{}.json", device='cuda', 
                    dtype='float16', seed=42, debug: bool = False):
    
    import t_search.datasets as datasets

    dtype_mapping = {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'bfloat16': torch.bfloat16
    }    

    dtype = dtype_mapping[dtype]

    kwargs = read_config(config)

    if not hasattr(datasets, dataset):
        raise ValueError(f"Unknown dataset {dataset}")
    
    rnd = np.random.default_rng(seed)
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(seed)

    ds: datasets.Benchmark = getattr(datasets, dataset)
    free_vars, target = ds.sample_set("train", device=device, dtype=dtype, generator=torch_gen, sorted=True)

    solver = GPSolver(
        device=device,
        dtype=dtype,
        rnd=rnd,
        torch_gen=torch_gen,
        debug=debug,
        **kwargs
    )

    solver.fit(free_vars, target)

    output = output.format(dataset)
    solver.save_metrics(output)

def args_pipeline():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='One of datasets from t_search.datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to config json file with specified constraints')
    parser.add_argument('--output', type=str, required=True, help='Output metrics file path pattern, use {} for dataset name')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='float16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    # parser.add_argument() 

    args = parser.parse_args()
    
    config_pipeline(
        dataset=args.dataset, 
        config=args.config, 
        output=args.output, 
        device=args.device, 
        dtype=args.dtype, 
        seed=args.seed,
        debug=args.debug
    )

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