from collections import deque
from dataclasses import dataclass, field
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
from t_search.operators.optim.term_hole import HoleFilling

from t_search.syntax import Term, TermPos
from t_search.syntax.term import Op, OptimPoint, Value
from t_search.utils import metrics_serializer

@dataclass(frozen=False)
class TermMutationContext: 
    gen: int
    term: Term 
    pos: TermPos
    pos_priority: tuple[float] | None = None
    pos_id: int = 0 
    num_pos: int = 0
    optim_term: Term | None = None 
    tabu_markers: list[Term] = field(default_factory=list)
    start_loss: float = 0.0
    optim_loss: float = 0.0
    num_minimas: int = 0
    # optim_vectors: list[torch.Tensor] = field(default_factory=list)
    num_optim_vectors: int = 0
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

@dataclass(frozen=True)
class L2BasedContinuation:
    filling: Term

# @dataclass(frozen=True)
# class TermMutationContextOrder: 
#     main_context: TermMutationContext
#     continuation: deque[LossBasedContinuation | L2BasedContinuation]

# @dataclass(order=True)
# class PrioritizedTermPos:
#     priority: Any 
#     optim_term: Term = field(compare=False)
#     pos: TermPos = field(compare=False)

# @dataclass(order=True)
# class OptimState:
#     priority: Any = field(compare=True)
#     optim_term: Term = field(compare=False) # term with OptimPoint
#     # optim_position: TermPos
#     term: Term = field(compare=False) # original term
#     position: TermPos = field(compare=False)
#     id: int = 0 
#     cnt: int = 0
#     # path: list[PathNode] # maps new pos to old pos, these points are also collected in optimization
#     # tabu_markers: set[Term] # set of skeletons that represent the optimization path
#     # binding: dict[OptimPoint, torch.Tensor]
#     # ranges: torch.Tensor
#     # const_binding: dict[OptimPoint, torch.Tensor] = field(default_factory=dict)

# @dataclass(order=True)
# class PrioritizedOptimStateGens:
#     optim_state: OptimState = field(compare=True)
#     optim_state_gen: Generator[OptimState, None, None] = field(compare=False)

# @dataclass(frozen=True)
# class Hole:    
#     start_loss: float # of parent term
#     term: Term # parent 
#     position: TermPos # where is the hole
#     bindings: list[torch.Tensor] # possible bindings for the hole

# @dataclass(frozen=False)
# class LogEntry: 
#     status: str = "active"
#     optim_term: Term | None = None 
#     optim_state: OptimState | None = None
#     optim_result: OptimResult | None = None
#     hole: Hole | None = None
#     fill_logs: list[HoleFillingLog] | None = None
            
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
                 remap_provider: Any,        
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
                 loss_threshold: float = 0.01,
                 log_file: str | None = None,
                 num_pos_per_term: int = 1,
                 children_limit: int = 10000,
                 loss_koef: float = 1.0,
                 best_by_metric: Literal["l2", "loss"] = "l2",
                #  debug: bool = False,
                #  rnd: np.random.Generator = GLOBAL_RNG,
                 **kwargs):
        super().__init__(**kwargs, rate=1.0)
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
        self.tried_optim_terms: set[Term] = set()
        self.tried_optim_terms_hit: int = 0
        self.lr = lr
        self.normalizer = normalizer
        self.max_evals = max_evals
        self.tolerance_change = tolerance_change
        self.tolerance_grad = tolerance_grad
        self.default_loss_fn = evaluator.get_loss_fn()
        self.num_minimas = num_minimas
        self.num_lib_terms = num_lib_terms
        self.loss_threshold = loss_threshold
        self.loss_koef = loss_koef
        self.init_op = init_op
        self.const_optimizer = const_optimizer
        self.num_pos_per_term = num_pos_per_term
        self.children_limit = children_limit
        self.best_by_metric = best_by_metric
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
        self.lineage: dict[Term, Term] = {} # child to parent map for backtracking
        self.term_contexts: dict[Term, deque[TermMutationContext]] = {} # cached position priorities for terms
        self.term_context_continuations: dict[tuple[Term, TermPos], deque[LossBasedContinuation | L2BasedContinuation]] = {} # current position order for term
        self.remap_provider = remap_provider
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
        if log_file is not None:
            self.log_file = open(log_file, "w")
        else:
            self.log_file = None
        self.mutation_log = []
        # self.debug = debug

    def init(self, evaluator, **_):

        # testing 
        # term = self.syntax.parse_term_str("(mul (add (mul (mul (add (mul (mul x0 x0) 1.0174) -2.0177) x0) x0) 1.0117) x0)")
        # res=self.const_optimizer.optimize(term, with_loss=True)
        # print(res)

        # end testing

        lib_terms = self.init_op() # these terms will be used for sketches
        terms = list(set(lib_terms))
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



    def rand_position_order(self, term: Term, positions: list[TermPos]) -> list[Any]:
        ''' Returns list of priorities for positions - (rand,) '''
        priorities = [(self.rnd.random(),) for _ in positions]
        return priorities

    def shallow_to_deep_position_order(self, term: Term, positions: list[TermPos]) -> list[Any]:
        priorities = [(p.at_depth, self.rnd.random(), ) for  p in positions]
        return priorities

    def best_grad_position_order(self, term: Term, positions: list[TermPos]) -> list[Any]:
        grads = get_all_grads(term, var_bindings=self.var_bindings, 
                      get_loss_fn=self.evaluator.get_loss_fn)
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

    def is_lincomb(self, pos: TermPos) -> bool:
        if pos.parent is None:
            return False
        term = pos.parent.term
        if isinstance(term, Op) and term.op_id in ["add", "mul"]:
            other_arg = term.get_args()[1 - pos.pos]
            if isinstance(other_arg, Value):
                return True 
        return False

    def _get_optim_state(self, context: TermMutationContext) -> None:
        ''' None is returned if term,position is already optimized '''

        if context.optim_term is not None:
            return # already computed for this context

        # while self.is_lincomb(position):
        #     position = position.parent

        # if position.parent is None:
        #     return None # cannot optimize root
        if self.is_lincomb(context.pos):
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

        context.tabu_markers = parent_optim_terms

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
            if self.debug:
                print(f"Tabu: {optim_term}")
            self.tabu_set.add(optim_term)

    def reduce_consts(self, term: Term, 
                        ops: dict[str, Callable] = {"add": lambda vs: sum(v for v in vs), "mul": lambda vs: prod(v for v in vs)},
                        identities: dict[str, Callable] = {}
                      ) -> Term:
        ''' add/mul for binary is tansformed to one of varying arity and then all constants are combined
            then, we return to binary ops. Top-down transfomration.
        '''
        if not isinstance(term, Op) or (term.op_id not in ops):
            return term
        all_terms = deque([term])
        final_args = []
        while len(all_terms) > 0:
            current = all_terms.popleft()
            if isinstance(current, Op) and (current.op_id == term.op_id):
                for a in current.get_args():                    
                    all_terms.append(a)
            else:
                reduced_current = self.reduce_consts(current, ops=ops, identities=identities)
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
        if (final_const is not None) and ((len(non_const_terms) == 0) or (term.op_id not in identities) or (not identities[term.op_id](final_const.value))):
            if term.op_id == "add": # insert const after (mul ...) when possible
                found_id = next((i for i, t in enumerate(non_const_terms) 
                                 if isinstance(t, Op) and t.op_id == "mul" and \
                                    any(isinstance(marg, Value) for marg in t.get_args())), None)
                if found_id is None:
                    non_const_terms.append(final_const)        
                else:    
                    # non_const_terms.insert(found_id + 1, final_const)
                    non_const_terms = [non_const_terms[found_id], final_const, *non_const_terms[:found_id], *non_const_terms[found_id + 1:]]
            else:
                non_const_terms.append(final_const)        
        if len(non_const_terms) == 1:
            return non_const_terms[0]
        new_term = self.syntax.get_op(term.op_id, *non_const_terms)
        return new_term
    
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
        new_term = self.reduce_consts(new_term)

        optimized = self.const_optimizer.optimize(new_term, with_loss=True)    
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
            num_pos = context.num_pos,
            optim_term = context.optim_term,
            tabu_markers = context.tabu_markers,
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

        self.term_contexts[context.term].appendleft(next_context)
    
    def mutate_one_position(self, term: Term, context: TermMutationContext) -> None:
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
            elif isinstance(next_cont, L2BasedContinuation):
                context.filling = next_cont.filling
                optimized = self.optimize_consts(context, context.filling)
                if optimized is not None:
                    context.final_term = optimized.term
                    context.final_loss = optimized.loss

            self.add_context_continuation(context)

            if context.final_term is None:
                context.status = "invalid_term"

            if context.final_loss < self.loss_koef * context.start_loss:
                self.num_better_fills += 1
            else:
                context.status = "no_better"

            return            
        
        optim_result: OptimResult = optimize_par(context.optim_term, 
                                self.ranges, 
                                {self.optim_point: self.semantics.get_outputs(context.pos.term)},
                                loss_fn_builder=self.evaluator.get_loss_fn,
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
        mask = loss_per_start >= self.loss_threshold
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

            el_diffs = optim_vectors.unsqueeze(1) - self.lib_vectors.unsqueeze(0) # optim_vecs, lib_vecs, dims
            all_dists = torch.sum(el_diffs ** 2, dim=-1) # optim_vecs, lib_vecs - note, maybe different distance measures
            best_term_vec_dists = torch.min(all_dists, dim=0).values # lib_vecs
            sort_ids = torch.argsort(best_term_vec_dists)
            sorted_dists = best_term_vec_dists[sort_ids]
            ordered_terms = [self.lib_terms[i] for i in sort_ids.tolist()]
            context.lib_term_dists = sorted_dists.tolist()
            context.lib_term_order = ordered_terms
            if len(ordered_terms) > self.num_lib_terms:
                ordered_terms = ordered_terms[:self.num_lib_terms]
                context.lib_term_dists = context.lib_term_dists[:self.num_lib_terms]
                context.lib_term_order = ordered_terms

        if self.best_by_metric == "loss": # run all optimizations of ordered terms abd pick best 
            final_losses = torch.full((len(ordered_terms),), torch.inf, dtype=self.target.dtype, device=self.target.device)    
            final_terms = []
            for i, term in enumerate(ordered_terms): 

                optimized = self.optimize_consts(context, term)

                if optimized is None:
                    final_terms.append(None)
                    continue

                final_losses[i] = optimized.loss

                final_terms.append(optimized.term)
                self.num_total_fills += 1
    
            sort_ids = torch.argsort(final_losses)
            context.final_losses = final_losses.tolist()
            best_loss_id = sort_ids[0].item()
            context.final_loss = final_losses[best_loss_id].item()
            context.final_term = final_terms[best_loss_id]     
            context.filling = ordered_terms[best_loss_id]   

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

        else: # best_by_metric == "l2"
            term = ordered_terms[0]
            context.filling = term
            optimized = self.optimize_consts(context, term)
            if optimized is None:
                context.final_loss = float('inf')
                context.final_term = None
            else:
                context.final_loss = optimized.loss
                context.final_term = optimized.term
            context.final_losses = [context.final_loss]

            conts = [L2BasedContinuation(ordered_terms[i]) for i in range(1, len(ordered_terms))]
            if len(conts) > 0:
                self.term_context_continuations[context_key] = deque(conts)

        self.add_context_continuation(context)

        if context.final_term is None:
            context.status = "invalid_term"

        if context.final_loss < self.loss_koef * context.start_loss:
            self.num_better_fills += 1
        else:
            context.status = "no_better"

        # self.mutation_log.append(context)
        return

        # return res_term
    
    def mutate_position(self, term: Term, contexts: list[TermMutationContext]) -> list[TermMutationContext] | None:
        self.mutation_log.extend(contexts)
        filtered_contexts = []
        for context in contexts:
            self.mutate_one_position(term, context)
            if context.status == "active":
                assert context.final_term is not None
                filtered_contexts.append(context)
        if len(filtered_contexts) == 0:
            return None
        filtered_contexts.sort(key=lambda c: c.final_loss)
        return filtered_contexts
       
    def select_positions(self, term: Term) -> Generator[list[TermMutationContext], None, None]:   

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
            # ppositions: list[TermMutationContext] = []
            # for p, pos in zip(priorities, positions):
            #     # while self.is_lincomb(pos):
            #     #     pos = pos.parent
            #     # if pos.parent is None:
            #     #     # return None # cannot optimize root                    
            #     #     continue
            #     # optim_term = self.syntax.replace_position(term, pos, self.optim_point, with_validation=False)
            #     ppos = PrioritizedTermPos(p, optim_term, pos) 
            #     ppositions.append(ppos)
            # ppositions.sort(key=lambda x: x.priority, reverse=True)
            self.term_contexts[term] = deque(contexts)
            
            # sorted_positions = [p for _, p in sorted(zip(priorities, positions), key=lambda x: x[0])]
            
        contexts = self.term_contexts[term]

        pos_id = 0
        num_pos = len(contexts)
        next_contexts = []
        while len(contexts) > 0:
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
                    continue

                # hole_pos = HolePos(priority, term, pos)
                # yield context
                next_contexts.append(context)
                if len(next_contexts) >= self.num_pos_per_term:
                    yield next_contexts
                    next_contexts = []
                # heappush(self.pos_queue, hole_pos)

            if len(next_contexts) > 0:
                yield next_contexts
                next_contexts = []

        visited = set()
        cur_term = term
        while len(self.term_contexts[cur_term]) == 0:
            if cur_term in visited:
                break
            visited.add(cur_term) 
            if cur_term in self.remap_provider.remap:
                term_key = self.remap_provider.remap[cur_term]
            else:
                term_key = cur_term   
            if term_key in self.lineage:                 
                cur_term = self.lineage[term_key]
            else:
                break
        if len(self.term_contexts[cur_term]) > 0:
            yield from self.select_positions(cur_term)         

        # if term in self.remap_provider.remap:
        #     term_key = self.remap_provider.remap[term]
        # else:
        #     term_key = term
        # if term_key in self.lineage: # we finished with term positions, backtrack to parent 
        #     parent_term = self.lineage[term_key]
        #     if parent_filling.hole_root != term:
        #         yield from self.select_positions(parent_filling.hole_root)

        # self.added_terms.add(term)
        return
    
    # def select_position(self, terms: list[Term]) -> Generator[OptimState, None, None]:
    #     ''' Builds one generator for optim states ordered by priority '''
    #     optim_state_cache = []
    #     for term in terms:
    #         # self.log.setdefault(term, [])

    #         term_gen = self.select_positions(term)
    #         cur_term = next(term_gen, None)
    #         if cur_term is not None:
    #             heappush(optim_state_cache, PrioritizedOptimStateGens(cur_term, term_gen))
    #     while len(optim_state_cache) > 0:
    #         cur = heappop(optim_state_cache)
    #         yield cur.optim_state
    #         next_optim_state = next(cur.optim_state_gen, None)
    #         if next_optim_state is not None:
    #             heappush(optim_state_cache, PrioritizedOptimStateGens(next_optim_state, cur.optim_state_gen))
    #     pass            
    
    # def __call__(self, population):        
    #     # delayed_fillings = self.term_hole_pairs.register_delayed_terms()
    #     all_fillings = []
    #     # self.log.clear()
    #     position_gen = self.select_position(population)
    #     count = 2*self.num_children # n per term 
    #     for optim_state in position_gen:
    #         new_fillings = self.mutate_position(optim_state)
    #         all_fillings.extend(new_fillings)
    #         if len(all_fillings) >= count:
    #             break

    #     # # selected_terms = set(f.found_term for f in all_fillings)
    #     # default_terms = self.term_hole_pairs.get_indexed_terms() 
    #     # # default_terms = [t for t in default_terms if t not in selected_terms]
    #     # # now for each term we would like to take the hole 
    #     # term_fillings = self.term_hole_pairs.find_holes_for_terms(default_terms, use_global_threshold=False)
    #     # all_fillings.extend(term_fillings)
    #     all_fillings.sort(key=lambda f: f.priority)
    #     selected_fillings = []
    #     present_terms = set()
    #     for filling in all_fillings:
    #         if filling.const_optim_term in present_terms:
    #             continue
    #         # pick best num_children fillings
    #         present_terms.add(filling.const_optim_term)
    #         selected_fillings.append(filling)
    #         if len(selected_fillings) >= self.num_children:
    #             break
    #         pass            
    #     children = []
    #     for f in selected_fillings:
    #         if f.term in self.lineage:
    #             prev_filling = self.lineage[f.term]
    #             if prev_filling.priority > f.priority:
    #                 self.lineage[f.term] = f
    #         else:
    #             self.lineage[f.term] = f # hole_root is parent term
    #         if self.debug:
    #             print(f)
    #         children.append(f.term)        
    #     return children

    def __call__(self, population):        
        self.mutation_log.clear()
        # parents = sorted(set(population), key=lambda t: self.syntax._get_term_priority(t))
        # tt = self.syntax.get_op("add", self.syntax.get_var("x0"), self.syntax.get_const(1.0))
        # self.evaluator.eval(tt)
        # parents.insert(0, tt)        
        mutations = super().__call__(population)
        new_children = []
        selected_contexts = []
        fixed_mutations = []
        for context in mutations:
            if not isinstance(context, Term):
            #     cur_term = context
            #     visited = set()
            #     while len(self.term_contexts[cur_term]) == 0:
            #         if cur_term in visited:
            #             break
            #         visited.add(cur_term) 
            #         if cur_term in self.remap_provider.remap:
            #             term_key = self.remap_provider.remap[cur_term]
            #         else:
            #             term_key = cur_term   
            #         if term_key in self.lineage:                 
            #             cur_term = self.lineage[term_key]
            #         else:
            #             break
            #     if len(self.term_contexts[cur_term]) > 0:
            #         new_children.append(cur_term)
            # else:
                fixed_mutations.append(context)
        should_exit = False
        for i in range(self.num_pos_per_term):
            num_added = 0
            for contexts in fixed_mutations:
                if i < len(contexts):
                    new_children.append(contexts[i].final_term)
                    selected_contexts.append(contexts[i])
                    self.lineage[contexts[i].final_term] = contexts[i].term
                    should_exit = len(new_children) >= self.children_limit
                    num_added += 1
                    if should_exit:
                        break
            should_exit = should_exit or num_added == 0
            if should_exit:
                break

        if self.log_file is not None:
            self.mutation_log.sort(key=lambda c: c.final_loss)
            for context in self.mutation_log:
                json.dump(context.__dict__, self.log_file, default=metrics_serializer)
                self.log_file.write("\n")
                self.log_file.flush()
        if self.debug:
            selected_contexts.sort(key=lambda c: c.final_loss)
            for context in selected_contexts:
                print(f"{context.final_loss:8.7f} ← {context.start_loss:8.7f} |\n\t\t{context.final_term} from\n\t\t{context.term}@({context.pos.term},{context.pos.occur}) with {context.filling}")
        # new_children = [ch for ch, p in zip(children, parents) if ch != p]
        # unique_children = sorted(set(new_children), key=lambda t: self.syntax._get_term_priority(t))
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
        