from collections import deque
from dataclasses import dataclass, field
from functools import partial
import json
from math import prod
from typing import Any, Callable, Generator, Literal

import torch

from t_search.base import ServiceBase
from t_search.evaluators.const_optimizer import ConstOptimizer, Optimized
from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.optimization import OptimResult, clean_optim_result, get_all_grads, optimize_par, set_local_minimas_, get_slowest_funs, optimize_seq
from t_search.evaluators.semantics import Semantics
from t_search.evaluators.term_spatial import Normalizer
from t_search.operators.initialization import Initialization
from t_search.operators.mutation import PositionMutation

from t_search.syntax import Term, TermPos
from t_search.syntax.term import Op, OptimPoint, Value
from t_search.utils import EvSearchTermination, metrics_serializer, rank, timed, unique_vector_ids, unique_vector_ids_batched

@dataclass(frozen=False)
class TermMutationContext: 
    gen: int
    term: Term 
    pos: TermPos
    pos_priority: tuple[float] | None = None
    pos_id: int = 0 
    num_pos: int = 0
    optim_term: Term | None = None 
    # tabu_markers: list[Term] = field(default_factory=list)
    start_loss: float | None = None
    optim_loss: float | None = None
    num_minimas: int | None = None
    # optim_vectors: list[torch.Tensor] = field(default_factory=list)
    num_optim_vectors: int = 0
    cont_id: int = 0
    found_const: float | None = None
    lib_term_dists: list[float] = field(default_factory=list)
    lib_term_order: list[Term] = field(default_factory=list)
    final_losses: list[float] = field(default_factory=list)
    final_loss: float = float('inf')
    final_term: Term | None = None
    filling: Term | None = None
    status: str = "active"


@dataclass(frozen=True)
class LossBasedContinuation:
    filling: Term
    final_term: Term
    final_loss: float

    def get_priority(self):
        return (self.final_loss,)

@dataclass(frozen=True)
class DistBasedContinuation:
    filling: Term
    filling_l2: float

    def get_priority(self):
        return (self.filling_l2,)
            
class PointOptim(PositionMutation, ServiceBase):
    ''' Position Optimization, adjust selected position with optimizer ''' 
    
    def __init__(self, *, 
                 var_bindings: dict[str, torch.Tensor],
                 target: torch.Tensor,
                 fitness: Fitness,
                #  syntax: Syntax,
                 semantics: Semantics,
                 evaluator: Evaluator,
                 torch_gen: torch.Generator,
                 get_cur_gen: Callable,
                 init_op: Initialization,
                 normalizer: Normalizer,
                 const_optimizer: ConstOptimizer,
                #  add_metrics: Callable, 
                #  target_variance: float = 1.0,          
                 position_strategy: Literal["rand_position_order", "shallow_to_deep_position_order", "best_grad_position_order"] = "rand_position_order",
                 num_starts: int = 10,
                 range_delta: float = 0.1,
                 max_evals: int = 20,
                 lr:float = 0.1,
                 tolerance_change: float = 1e-6,
                 tolerance_grad: float = 1e-3,
                 with_tabu: bool = True,
                 num_minimas: int = 1, # one hole can have multiple good bindings
                 num_lib_terms: int = 5,
                 pick_best_dists: bool = False,
                 loss_threshold: float = 0.01,
                 log_file: str | None = None,
                 loss_koef: float = 1.0,
                 best_by_metric: Literal["dist", "loss"] = "dist",
                 dist_measure: Literal["l2","pearson","spearman"] = "l2",
                 with_pop_terms: bool = False,
                 max_query_size: int = 1024,
                 max_query_depth: int = 2,
                 allow_no_better: bool = False,
                 backtrack_lineage: bool = True,
                 identity_atol: float = 0.001,
                 identity_rtol: float = 0.001,
                 with_reduction: bool = True,
                 with_subterms: bool = False,
                #  debug: bool = False,
                #  rnd: np.random.Generator = GLOBAL_RNG,
                 **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.evaluator = evaluator
        self.fitness = fitness
        # self.syntax = syntax
        self.semantics = semantics
        self.var_bindings = var_bindings
        self.with_tabu = with_tabu
        self.position_strategy = getattr(self, position_strategy)  
        # self.term_position_orders: dict[Term, deque] = {}
        # self.tabu_positions: dict[Term, set[TermPos]] = {} # any position below the tabu position should be ignored
        self.tabu_set: set[Term] = set()
        # self.add_metrics = add_metrics

        self.torch_gen = torch_gen
        self.num_starts = num_starts
        self.range_delta = range_delta
        self.with_pop_terms = with_pop_terms
        self.max_query_size = max_query_size
        self.tried_optim_terms: set[Term] = set()
        self.tried_optim_terms_hit: int = 0
        self.lr = lr
        self.normalizer = normalizer
        self.max_evals = max_evals
        self.tolerance_change = tolerance_change
        self.tolerance_grad = tolerance_grad
        self.num_minimas = num_minimas
        self.num_lib_terms = num_lib_terms
        self.pick_best_dists = pick_best_dists
        self.loss_threshold = loss_threshold
        self.loss_koef = loss_koef
        self.init_op = init_op
        self.const_optimizer = const_optimizer
        self.best_by_metric = best_by_metric
        self.dist_measure = dist_measure
        self.identity_atol = identity_atol
        self.identity_rtol = identity_rtol
        self.with_reduction = with_reduction
        self.max_query_depth = max_query_depth
        self.with_subterms = with_subterms
        # self.rnd=rnd

        # self.pos_queue: list[HolePos] = [] # hole priority queue
        # self.added_terms = set() # terms with added positions
        self.get_cur_gen = get_cur_gen
        min_y = torch.min(self.target) - self.range_delta
        max_y = torch.max(self.target) + self.range_delta
        self.ranges = torch.zeros((self.target.shape[0], 2), dtype=self.target.dtype, device=self.target.device)
        self.ranges[:, 0] = min_y
        self.ranges[:, 1] = max_y
        self.optim_point = OptimPoint(0)
        self.frontier: set[Term] = set() # current path ends
        self.deadends: set[Term] = set() # cannot improve anymore
        self.term_contexts: dict[Term, deque[TermMutationContext]] = {} # cached position priorities for terms
        self.term_context_continuations: dict[tuple[Term, TermPos], deque[LossBasedContinuation | DistBasedContinuation]] = {} # current position order for term
        self.allow_no_better = allow_no_better
        self.backtrack_lineage = backtrack_lineage
        # self.term_failed_contexts: list[TermMutationContext] = []
        # self.log: dict[Term, list[LogEntry]] = {} # stores current pop optimization log: wwhat positions tried, when and why they are failed

        # self.pool = 

        #TODO
        # 1. Metrics - DONE
        # 2. num_holes for one term - not just 1 - DONE
        # 2.2 num_bindings - DONE
        # 3. Const check for optimized hole. - DONE
        # 4. testing on simple term

        #metrics 
        self.num_better_fills = 0 
        self.num_total_fills = 0
        self.num_holes_created = 0
        self.num_terms_optimized = 0

        self.lib_terms = []
        self.lib_vectors = None
        self.query_terms = []
        self.query_vectors = None
        if log_file is not None:
            self.log_file = open(log_file, "w")
        else:
            self.log_file = None
        self.mutation_log = []
        self.mutation_log_per_term: dict[str, TermMutationContext] = {}
        # self.debug = debug

    def init(self, evaluator, **_):

        # testing 
        # term = self.syntax.parse_term_str("(mul (add (mul (mul (add (mul (mul x0 x0) 1.0174) -2.0177) x0) x0) 1.0117) x0)")
        # res=self.const_optimizer.optimize(term, with_loss=True)
        # print(res)

        # end testing

        lib_terms = self.init_op() # these terms will be used for sketches
        terms = sorted(set(lib_terms), key=lambda t: self.syntax._get_term_priority(t))
        evaluator.eval(terms) # compute unnormalized vectors 
        valid_terms = [t for t in terms if self.semantics.is_valid(t)]
        valid_terms.sort(key=lambda t: self.syntax._get_term_priority(t))
        vectors = self.semantics.get_outputs(valid_terms, return_type="list") # normalize vectors for better optimization performance
        final_terms = []
        final_vectors = []
        for term, vector in zip(valid_terms, vectors):
            if self.semantics.is_const(vector) is not None:
                continue
            final_terms.append(term)
            final_vectors.append(vector)
        # self.insert_terms(valid_terms, vectors)

        vectors = torch.stack(final_vectors, dim=0)

        normalized = self.normalizer.normalize(vectors)

        del vectors

        duplicate_mask = torch.isclose(normalized.unsqueeze(1), normalized.unsqueeze(0)).all(dim=-1)
        lower_tri = torch.tril(duplicate_mask, diagonal=-1)
        has_duplicate_before = lower_tri.any(dim=1)  # (n,) - True if vector i is duplicate of some j < i
        unique_mask = ~has_duplicate_before  # (n,) - True for unique vectors
        unique_indices = torch.where(unique_mask)[0]  # Indices of unique vectors
        final_terms = [final_terms[i] for i in unique_indices.tolist()]
        final_vectors = normalized[unique_indices]
        del normalized

        self.lib_terms = final_terms
        self.lib_vectors = final_vectors

        # target search 
        target_normalized = self.normalizer.get_normalized_target()
        close_mask = torch.isclose(final_vectors, target_normalized.unsqueeze(0)).all(dim=-1)
        close_ids = torch.where(close_mask)[0].tolist()
        attempted_solutions = []
        for close_id in close_ids:
            close_term = final_terms[close_id]
            optimized = self.optimize_lincomb(close_term)
            if optimized.loss < self.loss_threshold:
                attempted_solutions.append(optimized.term)
        if len(attempted_solutions)>0:
            self.evaluator.eval(attempted_solutions)
        pass

    def rand_position_order(self, term: Term, positions: list[TermPos]) -> list[Any]:
        ''' Returns list of priorities for positions - (rand,) '''
        priorities = [(self.rnd.random(),) for _ in positions]
        return priorities

    def shallow_to_deep_position_order(self, term: Term, positions: list[TermPos]) -> list[Any]:
        priorities = [(p.at_depth, self.rnd.random(), ) for  p in positions]
        return priorities

    def best_grad_position_order(self, term: Term, positions: list[TermPos]) -> list[Any]:
        grads = get_all_grads(term, var_bindings=self.var_bindings, 
                      get_loss_fn=partial(self.evaluator.get_loss_fn, with_mean_loss_logging=False))
        priorities = [ (-grads[(pos.term, pos.occur)], self.syntax.get_depth(pos.term)) for pos in positions]
        return priorities
    
    def optim_term_for_consts(self, term: Term) -> tuple[Term, dict[OptimPoint, torch.Tensor]]:
        const_optim_points: list[OptimPoint] = []
        const_binding = {}     

        # NOTE: taken from const_optimizer - should it be a separate routine?
        def const_to_optim_point(term, *_):
            if isinstance(term, Value):
                point_id = len(const_optim_points)
                point = OptimPoint(1 + point_id)
                const_optim_points.append(point)

                const_binding[point] = term.value
                return point

        optim_term = self.syntax.replace_fn(term, const_to_optim_point)      
        return (optim_term, const_binding)

    def is_in_lincomb(self, pos: TermPos) -> bool:
        if pos.parent is None:
            return False
        term = pos.parent.term
        if isinstance(term, Op) and term.op_id in ["add", "mul"]:
            other_arg = term.get_args()[1 - pos.pos]
            if isinstance(other_arg, Value):
                return True 
        return False

    # def is_lincomb(self, pos: TermPos) -> bool:
    #     if pos.parent is None:
    #         return False
    #     term = pos.parent.term
    #     if isinstance(term, Op) and term.op_id in ["add", "mul"]:
    #         other_arg = term.get_args()[1 - pos.pos]
    #         if isinstance(other_arg, Value):
    #             return True 
    #     return False

    def _get_optim_state(self, context: TermMutationContext) -> None:
        ''' None is returned if term,position is already optimized '''

        if context.optim_term is not None:
            return # already computed for this context

        if self.is_in_lincomb(context.pos):
            context.status = "skipped_lincomb"
            return None

        optim_term = self.syntax.replace_position(context.term, context.pos, self.optim_point, with_validation=False)
        context.optim_term = optim_term

        if optim_term in self.tried_optim_terms:
            self.tried_optim_terms_hit += 1
            context.status = "skipped_tried"
            # self.log[term][-1].status = "skipped_tried"
            # if self.debug:
            #     print(f"Skipped tried: {optim_term} for {term}@({position.term},{position.occur})")            
            return None
        self.tried_optim_terms.add(optim_term)

        # pos_outputs = self.semantics.get_outputs(position.term)

        # binding = { self.optim_point: pos_outputs }

        # range_mins = torch.minimum(pos_outputs, self.target)
        # range_maxs = torch.maximum(pos_outputs, self.target)
        # range_mins -= self.range_delta
        # range_maxs += self.range_delta        
        # ranges = torch.stack([range_mins, range_maxs], dim=0).t()

        optim_term_positions = self.syntax.get_positions(optim_term)
        optim_position = next(p for p in optim_term_positions if p.term == self.optim_point)
        # path: list[PathNode] = [] # excludes optim point and root
        # cur_pos_path = get_path(optim_position, with_current=False, with_root=False)
        # path = [PathNode(p) for p in cur_pos_path]

        parent_optim_terms = self.syntax.replace_path_unvalidated(optim_term, optim_position.parent, lambda *_: self.optim_point)
        if len(parent_optim_terms) > 0:
            parent_optim_terms.pop() # remove root OptimPoint

        # context.tabu_markers = parent_optim_terms

        if self.is_in_tabu(optim_term, parent_optim_terms):
            # self.log[term][-1].status = "tabu"
            context.status = "skipped_tabu"
            # if self.debug:
            #     print(f"Skipped tabu: {optim_term} for {term}@({position.term},{position.occur})")
            return None

        # tabu_markers = set()
        # if self.with_tabu: # creating tabu markers from path
        #     tabu_markers.add(optim_term)
        #     parent_optim_points = [self.optim_point for _ in path]
        #     if len(parent_optim_points) > 0:
        #         parent_optim_terms = self.syntax.replace_path_unvalidated(optim_term, optim_position.parent, parent_optim_points)
        #         for path_node, tabu_marker in zip(path, parent_optim_terms):
        #             path_node.tabu_marker = tabu_marker
        #         tabu_markers.update(parent_optim_terms)

        # optim_state = OptimState(priority, optim_term, term, position, id, cnt)
        # self.log[term][-1].optim_state = optim_state
        # return optim_state
        pass
    
    def is_in_tabu(self, optim_term: Term, parent_optim_terms) -> bool:
        if optim_term in self.tabu_set or any(t in self.tabu_set for t in parent_optim_terms):
            return True
        return False
    
    def add_to_tabu(self, optim_term: Term) -> None:
        if self.with_tabu:
            # if self.debug:
            #     print(f"Tabu: {optim_term}")
            self.tabu_set.add(optim_term)

    def decompose_lincomb(self, term: Term) -> tuple[Value | None, Term, Value | None]:
        ''' Returns (k, X, b) if term is of the form k * X + b or None otherwise. k, b can be None if they are not present. '''
        if isinstance(term, Op) and term.op_id == "add":
            args = term.get_args()
            if len(args) != 2:
                return None, term, None
            value_id = next((i for i, a in enumerate(args) if isinstance(a, Value)), None)
            if value_id is None:
                return None, term, None
            other_id = 1 - value_id
            b_value = args[value_id]
            other = args[other_id]
            if isinstance(other, Op) and other.op_id == "mul":
                mul_args = other.get_args()
                if len(mul_args) != 2:
                    return (None, other, b_value)
                k_id = next((i for i, a in enumerate(mul_args) if isinstance(a, Value)), None)
                if k_id is None:
                    return (None, other, b_value)
                X_id = 1 - k_id
                k_value = mul_args[k_id]
                X = mul_args[X_id]
                return (k_value, X, b_value)
            return (None, other, b_value)
        elif isinstance(term, Op) and term.op_id == "mul":
            args = term.get_args()
            if len(args) != 2:
                return None, term, None
            value_id = next((i for i, a in enumerate(args) if isinstance(a, Value)), None)
            if value_id is None:
                return None, term, None
            other_id = 1 - value_id
            k_value = args[value_id]
            X = args[other_id]
            return (k_value, X, None)
        return None, term, None

    def reduce_lincomb(self, term: Term, 
                        ops: dict[str, Callable] = {"add": lambda vs: sum(v for v in vs), "mul": lambda vs: prod(v for v in vs)},
                        identities: dict[str, Callable] = {}
                      ) -> Term:
        ''' add/mul for binary is tansformed to one of varying arity and then all constants are combined
            then, we return to binary ops. Top-down transformation.

            Additional rules are applied only when self.with_reduction.
            mul rule: (k1 * X + b1)*(k2 * X + b2) --> k3 * X^2 + k4 * X + b3 (4 to 3 consts)
            add rule: k1 * X + k2 * X --> k3 * X (2 to 1 consts, if X is the same)

            PROBLEMS: reduction leads to the following:
                  1. Removal of potential pathes to the target through new nodes 
                  2. Identity removal to some precision could fluctuate the loss especially when it is small enough  
                  3. A cicle in the lineage could appear 
            Pros: we target to reduce number of constants to optimize staying neutral (relativelly)
        '''
        if not isinstance(term, Op):
            return term
        if term.op_id not in ops:
            reduced_args = []
            for a in term.get_args():
                reduced_a = self.reduce_lincomb(a, ops=ops, identities=identities)
                reduced_args.append(reduced_a)
            new_term = self.syntax.get_op(term.op_id, *reduced_args)
            return new_term
        all_terms = deque([term])
        final_args = []
        while len(all_terms) > 0:
            current = all_terms.popleft()
            if isinstance(current, Op) and (current.op_id == term.op_id):
                for a in current.get_args():                    
                    all_terms.append(a)
            else:
                reduced_current = self.reduce_lincomb(current, ops=ops, identities=identities)
                if isinstance(reduced_current, Op) and (reduced_current.op_id == term.op_id):
                    for a in reduced_current.get_args():                    
                        all_terms.append(a) 
                else:               
                    final_args.append(reduced_current)
        const_terms, non_const_terms = [], []
        for a in final_args:
            (const_terms if isinstance(a, Value) else non_const_terms).append(a)
        # if len(const_terms) == 0: # nothing to reduce - leave as it was 
        #     return term
        # final_const = const_terms[0]
        final_const = None 
        if len(const_terms) == 1:
            final_const = const_terms[0]
        elif len(const_terms) > 1:
            reduce_fn = ops[term.op_id]
            new_const = reduce_fn([c.value for c in const_terms])
            final_const = new_const if isinstance(new_const, Value) else self.syntax.get_const(value=new_const)
        # if (final_const is not None) and ((len(non_const_terms) == 0) or (term.op_id not in identities) or (not identities[term.op_id](final_const.value))):
        add_identity_fn = identities.get("add", lambda _: False)
        mul_identity_fn = identities.get("mul", lambda _: False)
        if term.op_id == "add": # insert const after (mul ...) when possible

            if self.with_reduction:
                # rule 1: grouping ki * X in sum by X 
                decomposed_non_const_terms = [self.decompose_lincomb(t) for t in non_const_terms]
                groups = {}
                for (k,X,b) in decomposed_non_const_terms:
                    assert b is None
                    groups.setdefault(X, []).append((k or self.syntax.one_value).value)

                if len(groups) < len(decomposed_non_const_terms): # some grouped 
                    new_non_const_terms = []
                    for X, ks in groups.items():
                        k = sum(ks)
                        k_value = self.syntax.get_const(value=k)
                        if add_identity_fn(k_value.value): #mul 0
                            continue
                        if mul_identity_fn(k_value.value): #mul 1
                            new_X = X
                        else:
                            new_X = self.syntax.get_op("mul", k_value, X)
                        new_non_const_terms.append(new_X)
                    non_const_terms = new_non_const_terms

            if final_const is not None and not add_identity_fn(final_const.value): #add 0
                found_id = next((i for i, t in enumerate(non_const_terms) 
                                    if isinstance(t, Op) and t.op_id == "mul" and \
                                    any(isinstance(marg, Value) for marg in t.get_args())), None)
                if found_id is None:
                    non_const_terms.append(final_const)        
                else:    
                    # non_const_terms.insert(found_id + 1, final_const)
                    subterm = self.syntax.get_op("add", non_const_terms[found_id], final_const)
                    non_const_terms = [subterm, *non_const_terms[:found_id], *non_const_terms[found_id + 1:]]
            if len(non_const_terms) == 0:
                non_const_terms.append(self.syntax.zero_value)
        elif term.op_id == "mul": 
            if final_const is not None and not mul_identity_fn(final_const.value): #mul 1
                if self.with_reduction:
                    # rule 1: try to find (k * X + b) in args to add the constant there
                    decomposed_non_const_terms = [self.decompose_lincomb(t) for t in non_const_terms]                
                    found_id, k, X, b = next(((i, k, X, b) for i, (k, X, b) in enumerate(decomposed_non_const_terms) 
                                                if k is not None and b is not None), (None, None, None, None))
                    if found_id is not None:
                        new_k = self.syntax.get_const(value=k.value * final_const.value)
                        new_b = self.syntax.get_const(value=b.value * final_const.value)
                        if add_identity_fn(new_k.value): #mul 0
                            new_term = new_b
                        else:
                            if add_identity_fn(new_b.value): #add 0
                                new_b = None
                            if mul_identity_fn(new_k.value): #mul 1
                                new_k = None
                            if new_k is None and new_b is None:
                                new_term = X
                            elif new_k is None:
                                new_term = self.syntax.get_op("add", X, new_b)
                            elif new_b is None:
                                new_term = self.syntax.get_op("mul", new_k, X)
                            else:
                                new_term = self.syntax.get_op("add", self.syntax.get_op("mul", new_k, X), new_b)
                        non_const_terms[found_id] = new_term
                    else:
                        found_id, X, b = next(((i, X, b) for i, (k, X, b) in enumerate(decomposed_non_const_terms) 
                                                if k is None and b is not None), (None, None, None))
                        if found_id is not None:
                            new_k = final_const
                            new_b = self.syntax.get_const(value=b.value * final_const.value)
                            new_term = self.syntax.get_op("add", self.syntax.get_op("mul", new_k, X), new_b)
                            non_const_terms[found_id] = self.reduce_lincomb(new_term, ops=ops, identities=identities)
                        else:
                            non_const_terms.append(final_const)
                else:
                    non_const_terms.append(final_const)
            
            if len(non_const_terms) == 0:
                non_const_terms.append(self.syntax.one_value)                
        else:
            non_const_terms.append(final_const)      

        if len(non_const_terms) == 1:
            return non_const_terms[0]
        non_const_terms.sort(key=lambda t: self.syntax._get_term_priority(t), reverse=True)
        new_term = self.syntax.get_op(term.op_id, *non_const_terms)
        return new_term
    
    def reduce_identities(self, term, identities: dict[str, Value], atol=0.001, rtol=0.001) -> Term:
        if isinstance(term, Op) and term.op_id in identities:
            identity_value = identities[term.op_id]
            left_args = []
            orig_args = term.get_args()
            for arg in orig_args:
                if isinstance(arg, Value) and torch.isclose(arg.value, identity_value.value, atol=atol, rtol=rtol):
                    continue
                else:
                    left_args.append(self.reduce_identities(arg, identities, atol=atol, rtol=rtol))
            if len(left_args) == 0:
                return identity_value
            elif len(left_args) == 1:
                return left_args[0]
            # elif len(left_args) < len(orig_args):
            return self.syntax.get_op(term.op_id, *left_args)
            # return term
        elif isinstance(term, Op):
            new_args = [self.reduce_identities(arg, identities, atol=atol, rtol=rtol) for arg in term.get_args()]
            return self.syntax.get_op(term.op_id, *new_args)
        return term
                    
    def optimize_lincomb(self, term: Term) -> Optimized:
        if isinstance(term, Value):
            fit_term = term
        else:
            k_value = self.syntax.one_value
            b_value = self.syntax.zero_value
            fit_subterm1 = self.syntax.get_op("mul", k_value, term)
            fit_term = self.syntax.get_op("add", fit_subterm1, b_value)

        new_term = self.reduce_lincomb(fit_term)

        optimized = self.const_optimizer.optimize(new_term, with_loss=True)    

        return optimized

    
    def optimize_consts(self, context: TermMutationContext, term: Term) -> Optimized | None:
        if isinstance(term, Value):
            fit_term = term
        else:
            k_value = self.syntax.one_value
            b_value = self.syntax.zero_value
            fit_subterm1 = self.syntax.get_op("mul", k_value, term)
            fit_term = self.syntax.get_op("add", fit_subterm1, b_value)
        # NOTE: change hole pos to remove (mul ? c) and (add ? c).
        # TODO: check that next loop can be removed???
        # while hole_pos.parent is not None:   
        #     parent = hole_pos.parent             
        #     if isinstance(parent.term, Op) and \
        #         ((parent.term.op_id == "mul") or (parent.term.op_id == "add")) and \
        #         all(isinstance(a, Value) for arg_pos, a in enumerate(parent.term.get_args()) if arg_pos != hole_pos.pos):
        #         hole_pos = parent
        #         continue
        #     break                       

        # TODO: NOTE: here we do not handle const overflow 
        #             more precise handling woul try the term itself instead of linear combination
        #             when we above the const limit - for n ow it is intended. Just increase const limit in config (+2)
        new_term = self.syntax.replace_position(context.term, context.pos, fit_term)
        if new_term is None:
            # if self.debug:
            #     print(f"\tDiscarded: {fit_term} --> {hole_root}@({hole_pos.term}, {hole_pos.occur})")
            # status_setter("invalid_term")
            # final_terms.append(None)
            # continue
            return None
    
        # reducing consstants before optimization 
        new_term = self.reduce_lincomb(new_term)

        identity_reduced = False
        # reduced = True
        # reduced_term = new_term
        # reduced_cnt = 0
        # prev_optimized = None
        # while reduced:
        optimized1 = self.const_optimizer.optimize(new_term, with_loss=True) 

        if optimized1.loss <= self.fitness.fitness_atol:
            return optimized1
            # if optimized.loss < self.fitness.fitness_atol:
            #     break

            # if prev_optimized is not None and ((optimized.loss - prev_optimized.loss) / prev_optimized.loss) > 0.01: # reduction led to worse loss - stop
            #     optimized = prev_optimized
            #     break
            # prev_optimized = optimized
            # reduced = False
            # reduced_term = self.reduce_identities(optimized.term, {'add': self.syntax.zero_value, 'mul': self.syntax.one_value})
        def add_id_fn(v):
            nonlocal identity_reduced
            res = torch.isclose(v, self.syntax.zero_value.value, atol=self.identity_atol, rtol=self.identity_rtol)
            if res:
                identity_reduced = True
            return res
        def mul_id_fn(v):
            nonlocal identity_reduced
            res = torch.isclose(v, self.syntax.one_value.value, atol=self.identity_atol, rtol=self.identity_rtol)
            if res:
                identity_reduced = True
            return res
        reduced_term = self.reduce_lincomb(optimized1.term, identities={'add': add_id_fn, 'mul': mul_id_fn})                    
        if identity_reduced: # reduced_term != optimized1.term:
            optimized2 = self.const_optimizer.optimize(reduced_term, with_loss=True, num_starts=1, max_evals=1)
            optimized = optimized2
            # if optimized2.loss is None or ((optimized2.loss - optimized1.loss) / optimized1.loss > 0.01):
            #     optimized = optimized1
            # else:
            #     optimized = optimized2
        else:
            optimized = optimized1

        return optimized    
    
    def add_context_continuation(self, context: TermMutationContext) -> None:

        context_key = (context.term, context.pos)
        if context_key not in self.term_context_continuations:
            return 
        conts = self.term_context_continuations[context_key]
        if len(conts) == 0:
            del self.term_context_continuations[context_key]
            return

        next_context = TermMutationContext(
            gen = context.gen,
            term = context.term,
            pos = context.pos,
            pos_priority = context.pos_priority,
            pos_id = context.pos_id,
            cont_id = context.cont_id + 1,
            num_pos = context.num_pos,
            optim_term = context.optim_term,
            # tabu_markers = context.tabu_markers,
            start_loss = context.start_loss,
            optim_loss = context.optim_loss,
            num_minimas = context.num_minimas,
            num_optim_vectors = context.num_optim_vectors,
            found_const = context.found_const,
            lib_term_dists = context.lib_term_dists,
            lib_term_order = context.lib_term_order,
            # --- 
            final_losses = [],
            final_loss = float('inf'),
            final_term = None,
            filling = None,
            status = "active"            
        )

        self.term_contexts[context.term].append(next_context)
    
    def set_query(self, population: list[Term], max_term_depth = 3) -> None: 
        self.query_terms = self.lib_terms
        self.query_vectors = self.lib_vectors
        if not self.with_pop_terms:
            return
        present_pop_terms = set()
        pop_terms = []
        # pop_term_depth = self.syntax.max_term_depth - context.pos.at_depth
        for parent_term in population:
            parent_pos = self.syntax.get_positions(parent_term)
            for pos in parent_pos:
                if self.syntax.get_depth(pos.term) <= max_term_depth:
                    if isinstance(pos.term, Value) or pos.term in self.query_terms:
                        continue
                    if pos.term not in present_pop_terms:                        
                        present_pop_terms.add(pos.term)
                        pop_terms.append(pos.term)
            if parent_term not in present_pop_terms and self.syntax.get_depth(parent_term) <= max_term_depth and not isinstance(parent_term, Value):
                present_pop_terms.add(parent_term)
                pop_terms.append(parent_term)
        pass
        if len(pop_terms) == 0:
            return
        pop_terms = list(pop_terms)
        if len(pop_terms) > (self.max_query_size - len(self.query_terms)):
            self.rnd.shuffle(pop_terms) # we are going to pick self.max_query_size unique at max
        pop_vectors = self.semantics.get_outputs(pop_terms, return_type="tensor") 
        pop_normalized = self.normalizer.normalize(pop_vectors)
        del pop_vectors
        pop_terms = self.query_terms + pop_terms
        pop_query_vectors = torch.cat([self.query_vectors, pop_normalized], dim=0)
        del pop_normalized
        # batch_size = 1024
        # pop_duplicate_mask = torch.isclose(pop_query_vectors.unsqueeze(1), pop_query_vectors.unsqueeze(0)).all(dim=-1)
        # pop_lower_tri = torch.tril(pop_duplicate_mask, diagonal=-1)
        # del pop_duplicate_mask
        # pop_has_duplicate_before = pop_lower_tri.any(dim=1)  # (n,) - True if vector i is duplicate of some j < i
        # del pop_lower_tri
        # pop_unique_mask = ~pop_has_duplicate_before  # (n,) - True for unique vectors
        # del pop_has_duplicate_before
        # pop_unique_indices = torch.where(pop_unique_mask)[0]  # Indices of unique vectors
        pop_unique_indices = unique_vector_ids_batched(pop_query_vectors, batch_size=self.max_query_size, max_size=self.max_query_size)
        # del pop_unique_mask
        self.query_terms = [pop_terms[i] for i in pop_unique_indices.tolist()]
        self.query_vectors = pop_query_vectors[pop_unique_indices]      
        del pop_query_vectors, pop_unique_indices

    def finalize_context(self, context: TermMutationContext) -> None:
        
        if context.final_term is None:
            context.status = "invalid_term"
            return

        if context.final_loss < self.loss_koef * context.start_loss:
            # checking lineage for loop 
            if self.has_lineage_loop(context.final_term, context.term):
                context.status = "lineage_loop"
            if context.status == "active":
                self.num_better_fills += 1
        else:
            context.status = "no_better"

    def mutate_one_position(self, context: TermMutationContext) -> None:
        ''' Applies optimization to the term position '''
        
        # if self.debug: 
        #     print(f"\tOptim: {optim_state.optim_term}")

        context_key = (context.term, context.pos)
        if context_key in self.term_context_continuations: # it is continuation of optimization - we may reuse info from original optimization
            conts = self.term_context_continuations[context_key]
            assert len(conts) > 0
            next_cont = conts.popleft()
            if isinstance(next_cont, LossBasedContinuation):
                context.filling = next_cont.filling
                context.final_term = next_cont.final_term
                context.final_loss = next_cont.final_loss
            elif isinstance(next_cont, DistBasedContinuation):
                context.filling = next_cont.filling
                while True:
                    optimized = self.optimize_consts(context, context.filling)
                    if optimized is not None:
                        context.final_term = optimized.term
                        context.final_loss = optimized.loss
                        break
                    elif len(conts) == 0:
                        break 
                    else:
                        next_cont = conts.popleft()
                        context.filling = next_cont.filling
            else:
                raise ValueError(f"Unknown continuation type: {type(next_cont)}")

            # self.add_context_continuation(context)

            self.finalize_context(context)

            return            
        
        optim_result: OptimResult = optimize_par(context.optim_term, 
                                self.ranges, 
                                {self.optim_point: self.semantics.get_outputs(context.pos.term)},
                                loss_fn_builder=partial(self.evaluator.get_loss_fn, with_mean_loss_logging=False),
                                # pos_to_collect=pos_to_collect,
                                num_starts=self.num_starts,
                                lr=self.lr,
                                max_evals=self.max_evals,
                                tolerance_change=self.tolerance_change,
                                tolerance_grad=self.tolerance_grad,
                                torch_gen=self.torch_gen)

        # self.log[optim_state.term][-1].status = "optimized" if optim_result is not None else "no_minima"
        # self.log[optim_state.term][-1].optim_result = optim_result

        self.num_terms_optimized += 1
        loss_threshold = min(self.loss_threshold, context.start_loss)
        # loss_threshold = self.loss_threshold
        # if self.fitness.best_term_fitness is not None and self.fitness.best_term_fitness < loss_threshold:
        #     loss_threshold = self.fitness.best_term_fitness.item()

        # if optim_result is None:
        #     self.add_to_tabu(optim_state)
        #     return None                    

        # total_threshold_optim_result_(optim_result, self.loss_threshold)
        loss_per_start = optim_result.loss.mean(dim=-1)
        best_loss = torch.min(loss_per_start)
        context.optim_loss = best_loss.item()
        mask = loss_per_start >= loss_threshold
        optim_result.loss[mask, :] = torch.inf        

        if torch.any(torch.all(torch.isinf(optim_result.loss), dim=0)): # no minimas found
            self.add_to_tabu(context.optim_term)
            context.status = "no_minima"
            return None

        # holes: list[Hole] = []
            
        set_local_minimas_(optim_result)


        # TODO: add num of found minimas to context, add best minima loss 

        slowest_traces = get_slowest_funs(optim_result, max_num_funs=self.num_minimas, set_num_minimas=lambda n: setattr(context, "num_minimas", n))

        clean_optim_result(optim_result)

        # if slowest_traces is None:
        #     self.add_to_tabu(optim_state)
        #     continue
        
        # slowest_traces_binding = [t.clone() for traces in slowest_traces.binding.values() for t in traces] 

        optim_vectors = self.normalizer.normalize(list(slowest_traces.binding.values())[0])

        if len(optim_vectors) == 1 and torch.allclose(optim_vectors[0], self.normalizer.get_normalized_target()):
            # this vector does not make sense as it recursive search for target 
            self.add_to_tabu(context.optim_term)
            context.status = "recursive"
            return

        clean_optim_result(slowest_traces)

        context.num_optim_vectors = optim_vectors.shape[0]

        # search for constant vectors:
        consts = self.semantics.is_const(optim_vectors)
        possible_const = next((c for c in consts if c is not None), None)
        if possible_const is not None:
            context.found_const = possible_const
            context.status = "const"
            hole_term = self.syntax.get_const(value=possible_const)
            ordered_terms = [hole_term]
        else: # we order lib terms by dist measure

            should_clean_query = False
            if self.with_subterms:
                present_terms = set(self.query_terms)
                new_terms = []
                pos_max_depth = self.syntax.max_term_depth - context.pos.at_depth
                for subpos in self.syntax.get_positions(context.term):
                    if subpos.term not in present_terms and self.syntax.get_depth(subpos.term) <= pos_max_depth:
                        present_terms.add(subpos.term)
                        new_terms.append(subpos.term)
                if len(new_terms) > 0:
                    subterm_outputs = self.semantics.get_outputs(new_terms, return_type="tensor")
                    new_term_consts = self.semantics.is_const(subterm_outputs) # to set internal const cache
                    filtered_ids = [i for i, c in enumerate(new_term_consts) if c is None]
                    new_terms = [new_terms[i] for i in filtered_ids]
                    if len(new_terms) == 0:
                        query_vectors = self.query_vectors
                        query_terms = self.query_terms
                    else:
                        query_terms = self.query_terms + new_terms
                        filtered_subterm_outputs = subterm_outputs[filtered_ids]
                        del subterm_outputs
                        subterm_outputs = filtered_subterm_outputs
                        should_clean_query = True 
                        normalized_subterm_outputs = self.normalizer.normalize(subterm_outputs)
                        del subterm_outputs
                        query_vectors = torch.cat([self.query_vectors, normalized_subterm_outputs], dim=0)
                        del normalized_subterm_outputs  
                        uniq_ids = unique_vector_ids(query_vectors)
                        query_vectors = query_vectors[uniq_ids]
                        query_terms = [query_terms[i] for i in uniq_ids.tolist()]
                        del uniq_ids
                else:
                    query_vectors = self.query_vectors
                    query_terms = self.query_terms
            else:
                query_vectors = self.query_vectors
                query_terms = self.query_terms

            if self.dist_measure == "pearson":
                all_dists = torch.matmul(optim_vectors, query_vectors.t()) # optim_vecs, lib_vecs
                all_dists.neg_() # we want to maximize similarity, == minimize distance
            elif self.dist_measure == "spearman":
                optim_vectors_ranked = rank(optim_vectors)
                query_vectors_ranked = rank(query_vectors)
                optim_vectors_ranked_means = optim_vectors_ranked.mean(dim=1, keepdim=True)
                optim_vectors_ranked_centered = optim_vectors_ranked - optim_vectors_ranked_means
                optim_vectors_ranked_centered_normed = optim_vectors_ranked_centered / torch.norm(optim_vectors_ranked_centered, dim=1, keepdim=True).clamp(min=1e-7)
                query_vectors_ranked_means = query_vectors_ranked.mean(dim=1, keepdim=True)
                query_vectors_ranked_centered = query_vectors_ranked - query_vectors_ranked_means   
                query_vectors_ranked_centered_normed = query_vectors_ranked_centered / torch.norm(query_vectors_ranked_centered, dim=1, keepdim=True).clamp(min=1e-7)
                all_dists = torch.matmul(optim_vectors_ranked_centered_normed, query_vectors_ranked_centered_normed.t())
                all_dists.neg_()
                del optim_vectors_ranked, query_vectors_ranked
            elif self.dist_measure == "l2":
                el_diffs = optim_vectors.unsqueeze(1) - query_vectors.unsqueeze(0) # optim_vecs, lib_vecs, dims
                all_dists = torch.sum(el_diffs ** 2, dim=-1) # optim_vecs, lib_vecs - note, maybe different distance measures
            else:
                raise ValueError(f"Unknown dist measure: {self.dist_measure}")
            
            if should_clean_query:
                del query_vectors
            
            # depth_mask = [self.syntax.get_depth(t) + context.pos.at_depth <= self.syntax.max for t in query_terms]

            best_term_vec_dists = torch.min(all_dists, dim=0).values # lib_vecs
            del all_dists
            sort_ids = torch.argsort(best_term_vec_dists)
            sorted_dists = best_term_vec_dists[sort_ids]
            ordered_terms = [query_terms[i] for i in sort_ids.tolist()]
            context.lib_term_dists = sorted_dists.tolist()
            context.lib_term_order = ordered_terms
            del best_term_vec_dists, sort_ids, sorted_dists
            # if len(ordered_terms) > self.num_lib_terms:
            #     ordered_terms = ordered_terms[:self.num_lib_terms]
            #     context.lib_term_dists = context.lib_term_dists[:self.num_lib_terms]
            #     context.lib_term_order = ordered_terms

        best_dist = context.lib_term_dists[0]

        if self.best_by_metric == "loss": # run all optimizations of ordered terms abd pick best 
            sz = len(ordered_terms) if self.pick_best_dists else min(self.num_lib_terms, len(ordered_terms))
            final_losses = torch.full((sz,), torch.inf, dtype=self.target.dtype, device=self.target.device)    
            final_terms = []
            final_fillings = []
            lib_term_dists = []
            for term, dist in zip(ordered_terms, context.lib_term_dists): 
                i = len(final_terms)
                if (i >= sz) or (self.pick_best_dists and dist > best_dist):
                    break

                optimized = self.optimize_consts(context, term)

                if optimized is None:
                    # final_terms.append(None)
                    continue
                
                final_losses[i] = optimized.loss
                final_terms.append(optimized.term)
                final_fillings.append(term)
                lib_term_dists.append(dist)
                self.num_total_fills += 1

            final_losses = final_losses[:len(final_terms)]
            
            # while len(final_terms) < sz:
            #     final_terms.append(None)
            #     lib_term_dists.append(float('inf'))

            context.lib_term_dists = lib_term_dists
            context.lib_term_order = final_terms
    
            sort_ids = torch.argsort(final_losses)
            context.final_losses = final_losses.tolist()
            best_loss_id = sort_ids[0].item()
            context.final_loss = final_losses[best_loss_id].item()
            context.final_term = final_terms[best_loss_id]     
            context.filling = final_fillings[best_loss_id]   

            conts = []
            for i in range(1, len(sort_ids)):
                term_id = sort_ids[i].item()
                filling = ordered_terms[term_id]
                loss = final_losses[term_id].item()
                final_term = final_terms[term_id]
                cont = LossBasedContinuation(filling=filling, final_term=final_term, final_loss=loss)
                conts.append(cont)

            if len(conts) > 0:
                self.term_context_continuations[context_key] = deque(conts)                

        elif self.best_by_metric == "dist":
            # final_term = ordered_terms[0]
            context.final_loss = float('inf')
            context.final_term = None    
            next_i = 0   
            for final_term in ordered_terms:
                next_i += 1
                context.filling = final_term
                optimized = self.optimize_consts(context, context.filling)
                if optimized is not None:
                    context.final_loss = optimized.loss
                    context.final_term = optimized.term
                    break
            context.final_losses = [context.final_loss]

            if self.pick_best_dists:
                conts = []
                start_i = next_i
                while next_i < len(ordered_terms):
                    cur_dist = context.lib_term_dists[next_i]
                    if cur_dist > best_dist:
                        break
                    next_i += 1
                index_range = range(start_i, next_i)
                end_index = next_i
            else: 
                end_index = min(len(ordered_terms), next_i + self.num_lib_terms - 1)
                index_range = range(next_i, end_index)

            conts = [DistBasedContinuation(ordered_terms[i], context.lib_term_dists[i]) for i in index_range]
            context.lib_term_order = ordered_terms[:end_index]
            context.lib_term_dists = context.lib_term_dists[:end_index]
            if len(conts) > 0:
                self.term_context_continuations[context_key] = deque(conts)
        else:
            raise ValueError(f"Unknown best_by_metric: {self.best_by_metric}")

        # self.add_context_continuation(context)

        self.finalize_context(context)

        # self.mutation_log.append(context)
        return

        # return res_term
    
    def mutate_position(self, _: Term, context: TermMutationContext) -> TermMutationContext | None:
        self.mutation_log.append(context)
        self.mutation_log_per_term.setdefault(context.term, []).append(context)
        self.mutate_one_position(context)
        if (context.status == "active") or (self.allow_no_better and context.status == "no_better"):
            assert context.final_term is not None
            return context
        else: # mutation fail - add back the context continuations
            self.add_context_continuation(context)
            return None 
       
    def select_positions(self, term: Term) -> Generator[TermMutationContext, None, None]:   

        if term not in self.term_contexts:
            if not self.semantics.is_valid(term): # do not optimize invalid terms
                return
            cur_gen = self.get_cur_gen()
            positions = self.syntax.get_positions(term)
            priorities = self.position_strategy(term, positions) # should be cached if necessary
            start_loss = self.fitness.get_fitness(term).item()
            contexts = [TermMutationContext(cur_gen, term, pos, pos_priority=priority, start_loss=start_loss) 
                        for priority, pos in zip(priorities, positions) ]
            contexts.sort(key=lambda x: x.pos_priority)

            self.term_contexts[term] = deque(contexts)
            
            # sorted_positions = [p for _, p in sorted(zip(priorities, positions), key=lambda x: x[0])]
            
        contexts = self.term_contexts[term]

        pos_id = 0
        num_pos = len(contexts) 

        while len(contexts) > 0:
            context = contexts.popleft()
            context.pos_id = pos_id
            context.num_pos = num_pos
            pos_id += 1
            # log_entry = LogEntry(optim_term=ppos.optim_term)
            # self.log[term].append(log_entry)
            self._get_optim_state(context)            
            if context.status != "active":
                self.mutation_log.append(context)
                self.mutation_log_per_term.setdefault(context.term, []).append(context)
                continue

            # hole_pos = HolePos(priority, term, pos)
            yield context
            # heappush(self.pos_queue, hole_pos)

        # no more positions - all attempts failed - adding failed contexts back and try backtrack
        # for context in self.term_failed_contexts:
        #     self.add_context_continuation(context)
        # self.term_failed_contexts.clear()

        backtrack_term = self.get_backtrack_term(term)
        if backtrack_term is not None:
            yield from self.select_positions(backtrack_term)         

        return
    
    def mutate_term(self, term):
        base_fn = super().mutate_term
        def fn():
            # self.term_failed_contexts.clear()
            res = base_fn(term)
            # for failed_context in self.term_failed_contexts:
            #     self.add_context_continuation(failed_context)
            # self.term_failed_contexts.clear()
            return res        
        # if self.debug:
        #     res, time = timed(fn)()
        #     print(f"{time}ms: {term}")
        # else:            
        res = fn()
        return res

    def add_to_lineage(self, parent: Term, child: TermMutationContext):
        if child.final_term is not None:
            self.term_lineage.setdefault(child.final_term, []).append(parent)

    def get_backtrack_term(self, term: Term) -> Term | None:
        if not self.backtrack_lineage:
            return None
        cur_lineage = self.get_term_history(term)
        filtered_lineage = [fp for cur_terms in cur_lineage 
                                for fp in [[t for t in cur_terms 
                                                if t in self.term_contexts and len(self.term_contexts[t]) > 0]]
                                if len(fp) > 0]
        if len(filtered_lineage) > 0:           
            backtrack_terms = filtered_lineage[0] # parent, or random
            parent_term = self.rnd.choice(backtrack_terms) # go to random parent
            return parent_term
        return None

    def __call__(self, population):        
        self.mutation_log.clear()
        # parents = sorted(set(population), key=lambda t: self.syntax._get_term_priority(t))
        # tt = self.syntax.get_op("add", self.syntax.get_var("x0"), self.syntax.get_const(1.0))
        # self.evaluator.eval(tt)
        # parents.insert(0, tt)        
        self.set_query(population, max_term_depth=self.max_query_depth)
        mutations = super().__call__(population)
        if self.with_pop_terms:
            del self.query_vectors
        new_children = []
        # selected_contexts = []
        # fixed_mutations = []
        # retry_terms = []
        for context in mutations:
            if isinstance(context, Term): # retry term
                backtrack_term = self.get_backtrack_term(context)
                if backtrack_term is not None:
                    new_children.append(backtrack_term)
                else:
                    # if self.debug:
                    #     print(f"Done {cur_term}")
                    pass
            else: 
                self.lineage[context.final_term] = context.term
                self.frontier.add(context.final_term)
                self.frontier.discard(context.term)
                new_children.append(context.final_term)
        # should_exit = False
        # for i in range(self.num_pos_per_term):
        #     num_added = 0
        #     for contexts in fixed_mutations:
        #         if i < len(contexts):
        #             new_children.append(contexts[i].final_term)
        #             selected_contexts.append(contexts[i])
        #             self.lineage[contexts[i].final_term] = contexts[i].term
        #             should_exit = len(new_children) >= self.children_limit
        #             num_added += 1
        #             if should_exit:
        #                 break
        #     should_exit = should_exit or num_added == 0
        #     if should_exit:
        #         break

        # left_children = self.children_limit - len(new_children)
        # if left_children > 0 and len(retry_terms) > 0:
        #     if len(retry_terms) > left_children:
        #         retry_terms = self.rnd.choice(retry_terms, size=left_children, replace=False).tolist()
        #     new_children.extend(retry_terms)

        if self.log_file is not None:
            self.mutation_log.sort(key=lambda c: c.final_loss)
            for context in self.mutation_log:
                json.dump(context.__dict__, self.log_file, default=metrics_serializer)
                self.log_file.write("\n")
                self.log_file.flush()
        # if self.debug:
        #     selected_contexts.sort(key=lambda c: c.final_loss)
        #     for context in selected_contexts:
        #         # print(f"{context.final_loss:8.7f} ← {context.start_loss:8.7f} |\n\t\t{context.final_term} from\n\t\t{context.term}@({context.pos.term},{context.pos.occur}) with {context.filling}")
        #         print(f"{context.final_loss:8.7f} ← {context.start_loss:8.7f} | {context.final_term}")
        # new_children = [ch for ch, p in zip(children, parents) if ch != p]
        # unique_children = sorted(set(new_children), key=lambda t: self.syntax._get_term_priority(t))
        if len(new_children) == 0: # deadends for parents 
            for term in population:
                self.deadends.add(term)
            new_children = list(set.difference(self.frontier, self.deadends))
            pass
        if len(new_children) == 0:
            raise EvSearchTermination("DEADEND")
        return new_children
    
    def get_finalizer(self):
        self.add_metrics(
            num_better_fills=self.num_better_fills,
            num_total_fills=self.num_total_fills,
            num_holes_created=self.num_holes_created,
            num_terms_optimized=self.num_terms_optimized,
            tried_optim_terms=len(self.tried_optim_terms),
            tried_optim_terms_hit=self.tried_optim_terms_hit,
            # "num_terms_created": self.num_terms_created,
            # tabu_positions=sum(len(v) for v in self.tabu_positions.values())
            tabu_positions=len(self.tabu_set)
            )
        def finalizer():
            self.log_file and self.log_file.close()
        return finalizer
        