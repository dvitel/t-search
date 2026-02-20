from collections import deque
from dataclasses import dataclass, field
import json
import math
from typing import Any, Callable, Generator, Literal

from pyparsing import Sequence
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

from t_search.base import ServiceBase
from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.optimization import OptimResult, clean_optim_result, get_all_grads, optimize_par, set_local_minimas_, get_slowest_funs
from t_search.evaluators.semantics import Semantics
from t_search.evaluators.term_spatial import Normalizer
from t_search.operators.initialization import Initialization
from t_search.operators.mutation import PositionMutation

from t_search.operators.reduction import LincombMixin
from t_search.syntax import Term, TermPos
from t_search.syntax.term import OptimPoint, Value
from t_search.utils import EvSearchTermination, metrics_serializer, unique_vector_ids_seq

@dataclass(frozen=False)
class TermMutationContext: 
    gen: int
    term: Term 
    pos: TermPos
    stripped_pos_term: Term # without lincomb
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
    dist: float | None = None
    # lib_term_dists: list[float] = field(default_factory=list)
    # lib_term_order: list[Term] = field(default_factory=list)
    final_losses: list[float] = field(default_factory=list)
    final_loss: float = float('inf')
    final_term: Term | None = None
    filling: Term | None = None
    status: str = "active"
    reattempt: bool = False


@dataclass(frozen=True)
class LossBasedContinuation:
    context: TermMutationContext # original cont_id = 0 context
    filling: Term
    final_term: Term
    final_loss: float
    dist: float

    def get_priority(self):
        return (self.final_loss, self.dist,)

@dataclass(frozen=True)
class DistBasedContinuation:
    context: TermMutationContext # original cont_id = 0 context
    dist: float
    k: float 
    stripped_filling: Term
    b: float

    def get_priority(self):
        return (self.dist,)
            
class PointOptim(PositionMutation, ServiceBase, LincombMixin):
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
                 num_conts: int = 1000,
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
                 backtrack: Literal["lineage", "worse_fillings", "rand", "none"] = "lineage",
                 backtrack_rand_grow_depth: int = 6,
                 num_worse_fillings: int | None = None,
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
        self.num_conts = num_conts
        self.pick_best_dists = pick_best_dists
        self.loss_threshold = loss_threshold
        self.loss_koef = loss_koef
        self.init_op = init_op
        self.best_by_metric = best_by_metric
        self.dist_measure = dist_measure
        self.identity_atol = identity_atol
        self.identity_rtol = identity_rtol
        self.with_reduction = with_reduction
        self.max_query_depth = max_query_depth
        self.with_subterms = with_subterms

        # next is for computing correlation
        self.stats_loss_improvement = []
        self.stats_dists = []
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
        self.deadends: set[Term] = set() # cannot improve anymore
        self.term_contexts: dict[Term, deque[TermMutationContext]] = {} # cached position priorities for terms
        self.term_continuations: dict[Term, list[LossBasedContinuation | DistBasedContinuation]] = {} # current position order for term
        self.allow_no_better = allow_no_better
        self.backtrack = backtrack if backtrack != "none" else None
        self.backtrack_rand_grow_depth = backtrack_rand_grow_depth
        self.num_worse_fillings = num_worse_fillings or 0
        self.worse_filling_counts: dict[Term, int] = {}
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

    def init(self):

        lib_terms = self.init_op() # these terms will be used for sketches
        terms = sorted(set(lib_terms), key=lambda t: self.syntax._get_term_priority(t))
        self.evaluator.eval(terms) # compute unnormalized vectors 
        final_terms = [t for t in terms if self.semantics.is_valid(t)]
        final_terms.sort(key=lambda t: self.syntax._get_term_priority(t))
        final_vectors = self.semantics.get_outputs(final_terms, return_type="tensor") # normalize vectors for better optimization performance
        # final_terms = []
        # final_vectors = []
        # for term, vector in zip(valid_terms, vectors):
        #     if self.semantics.is_const(vector) is not None:
        #         continue
        #     final_terms.append(term)
        #     final_vectors.append(vector)
        # self.insert_terms(valid_terms, vectors)

        # vectors = torch.stack(final_vectors, dim=0)

        normalized = self.normalizer.normalize(final_vectors)

        del final_vectors

        unique_indices = unique_vector_ids_seq(normalized)
        final_terms = [final_terms[i] for i in unique_indices]
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
            new_kb, *_ = self.optimize_lincomb([close_term], self.target.unsqueeze(0))
            if new_kb[0] < self.loss_threshold:
                new_term = self.build_lincomb(new_kb[1], new_kb[2], new_kb[3])
                attempted_solutions.append(new_term)
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
                      get_loss_fn=self.evaluator.get_loss_fn)
        # pos_by_depth = [[] for _ in range(self.syntax.get_depth(term)+1)] # lists of poss 
        # for pos_i, pos in enumerate(positions):
        #     pos_priority = (-grads[(pos.term, pos.occur)], self.syntax._get_term_priority(pos.term))
        #     pos_by_depth[pos.at_depth].append((pos_priority, pos_i))
        # for poss in pos_by_depth:
        #     poss.sort(key=lambda p: -torch.norm(grads[(p.term, p.occur)]).item())
        priorities = [ (-grads[(pos.term, pos.occur)], self.syntax.get_depth(pos.term)) for pos in positions]
        return priorities
    
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
    
    def add_context_continuation(self, cont: DistBasedContinuation | LossBasedContinuation, cont_id: int) -> None:

        old_context = cont.context

        context = TermMutationContext(
            gen = old_context.gen,
            term = old_context.term,
            pos = old_context.pos,
            stripped_pos_term= old_context.stripped_pos_term,
            pos_priority = old_context.pos_priority,
            pos_id = old_context.pos_id,
            cont_id = cont_id,
            num_pos = old_context.num_pos,
            optim_term = old_context.optim_term,
            # tabu_markers = context.tabu_markers,
            start_loss = old_context.start_loss,
            optim_loss = old_context.optim_loss,
            num_minimas = old_context.num_minimas,
            num_optim_vectors = old_context.num_optim_vectors,
            found_const = old_context.found_const,            
            # lib_term_dists = old_context.lib_term_dists,
            # lib_term_order = old_context.lib_term_order,
            # --- 
            final_losses = [],
            final_loss = float('inf'),
            dist = cont.dist,
            final_term = None,
            filling = None,
            status = "active"            
        )

        if isinstance(cont, LossBasedContinuation):
            context.filling = cont.filling
            context.final_term = cont.final_term
            context.final_loss = cont.final_loss
            context.dist = cont.dist
        elif isinstance(cont, DistBasedContinuation):
            # if not (cont.stripped_filling == context.stripped_pos_term):
            filling = self.build_lincomb(cont.k, cont.stripped_filling, cont.b)
            if not (isinstance(filling, Value) and isinstance(context.pos.term, Value) and torch.isclose(filling.value, context.pos.term.value, atol=self.identity_atol, rtol=self.identity_rtol)):
                context.filling = filling
        if context.filling is not None:
            self.term_contexts[context.term].append(context)
    
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
        pop_unique_indices = unique_vector_ids_seq(pop_query_vectors) # batch_size=self.max_query_size, max_size=self.max_query_size)
        # del pop_unique_mask
        self.query_terms = [pop_terms[i] for i in pop_unique_indices]
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
    
    def combine_new_term(self, root: Term, pos: TermPos, filling: Term) -> Term | None:

        new_term = self.syntax.replace_position(root, pos, filling)
        if new_term is None:
            return None
    
        # reducing consstants before optimization 
        # new_term = self.reduce_lincomb(new_term)

        reduced_term = self.reduce_lincomb(new_term, identities={'add': self.add_id_fn, 'mul': self.mul_id_fn})   

        if not self.syntax.is_valid(reduced_term):
            return new_term

        return reduced_term    

    def mutate_one_position(self, context: TermMutationContext) -> None:
        ''' Applies optimization to the term position '''
        
        # if self.debug: 
        #     print(f"\tOptim: {optim_state.optim_term}")

        if context.cont_id > 0: # it is continuation of optimization - we may reuse info from original optimization
            assert context.filling is not None
            if context.final_term is None:
                new_term = self.combine_new_term(context.term, context.pos, context.filling)
                if new_term is not None:
                    self.evaluator.eval(new_term)
                    loss = self.fitness.get_fitness(new_term).item()
                    context.final_term = new_term
                    context.final_loss = loss    
                    assert context.dist is not None                               

            self.finalize_context(context)

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
        loss_threshold = min(self.loss_threshold, context.start_loss)
        # loss_threshold = self.loss_threshold
        # if self.fitness.best_term_fitness is not None and self.fitness.best_term_fitness < loss_threshold:
        #     loss_threshold = self.fitness.best_term_fitness.item()

        # if optim_result is None:
        #     self.add_to_tabu(optim_state)
        #     return None                    

        # total_threshold_optim_result_(optim_result, self.loss_threshold)
        num_better_tests_per_start = (optim_result.loss < loss_threshold).sum(dim=-1)
        loss_per_start = optim_result.loss.mean(dim=-1)
        best_loss = torch.min(loss_per_start)
        context.optim_loss = best_loss.item()
        mask = num_better_tests_per_start < 0.75 * optim_result.loss.shape[-1]
        optim_result.loss[mask, :] = torch.inf        

        if torch.any(torch.all(torch.isinf(optim_result.loss), dim=0)): # no minimas found
            self.add_to_tabu(context.optim_term)
            context.status = "no_minima"
            return None

        set_local_minimas_(optim_result)

        slowest_traces = get_slowest_funs(optim_result, max_num_funs=self.num_minimas, set_num_minimas=lambda n: setattr(context, "num_minimas", n))

        clean_optim_result(optim_result)

        # if slowest_traces is None:
        #     self.add_to_tabu(optim_state)
        #     continue
        
        # slowest_traces_binding = [t.clone() for traces in slowest_traces.binding.values() for t in traces] 

        optim_vectors_plain = list(slowest_traces.binding.values())[0]
        optim_vectors = self.normalizer.normalize(optim_vectors_plain)

        if len(optim_vectors) == 1 and torch.allclose(optim_vectors[0], self.normalizer.get_normalized_target()):
            # this vector does not make sense as it recursive search for target 
            self.add_to_tabu(context.optim_term)
            context.status = "recursive"
            return        

        context.num_optim_vectors = optim_vectors.shape[0]

        # search for constant vectors:
        consts = self.semantics.is_const(optim_vectors)
        possible_const = next((c for c in consts if c is not None), None)
        if possible_const is not None:
            context.found_const = possible_const
            context.status = "const"
            ordered_kb = [((-optim_vectors.shape[-1], 0.0), 0.0, self.syntax.zero_value, possible_const)]
        else: # we order lib terms by dist measure            

            # pos_k, pos_subterm, pos_b = self.decompose_lincomb(context.pos.term)

            # if pos_b is not None: 
            #     prohibited_term = pos_subterm # cannot use it at place

            query_terms = self.query_terms
            if self.with_subterms:
                present_terms = set(self.query_terms)
                new_terms = []
                # pos_max_depth = min(self.syntax.max_term_depth - context.pos.at_depth, self.max_query_depth)
                pos_max_depth = self.syntax.get_depth(context.pos.term) - 2 # for lincomb
                for subpos in self.syntax.get_positions(context.term):
                    if subpos.term not in present_terms and self.syntax.get_depth(subpos.term) <= pos_max_depth:
                        present_terms.add(subpos.term)
                        new_terms.append(subpos.term)
                if len(new_terms) > 0:
                    subterm_outputs = self.semantics.get_outputs(new_terms, return_type="tensor")
                    new_term_consts = self.semantics.is_const(subterm_outputs) # to set internal const cache
                    filtered_ids = [i for i, c in enumerate(new_term_consts) if c is None]
                    new_terms = [new_terms[i] for i in filtered_ids]
                    if len(new_terms) > 0:
                        query_terms = self.query_terms + new_terms
                        filtered_subterm_outputs = subterm_outputs[filtered_ids]
                        del subterm_outputs
                        subterm_outputs = filtered_subterm_outputs
                        normalized_subterm_outputs = self.normalizer.normalize(subterm_outputs)
                        del subterm_outputs
                        query_vectors = torch.cat([self.query_vectors, normalized_subterm_outputs], dim=0)
                        del normalized_subterm_outputs  
                        uniq_ids = unique_vector_ids_seq(query_vectors)
                        del query_vectors
                        query_terms = [query_terms[i] for i in uniq_ids]
                        del uniq_ids

            ordered_kb = self.optimize_lincomb_batched(query_terms, optim_vectors_plain, iters=128,
                                                        candidates_batch=64 if self.semantics.dims >= 100 else 256,
                                                        targets_batch=16)

            del optim_vectors
            clean_optim_result(slowest_traces)

            # context.lib_term_dists = list(result_dists)
            # context.lib_term_order = list(result_terms)

            # if self.dist_measure == "pearson":
            #     all_dists = torch.matmul(optim_vectors, query_vectors.t()) # optim_vecs, lib_vecs
            #     all_dists.neg_() # we want to maximize similarity, == minimize distance
            # elif self.dist_measure == "spearman":
            #     optim_vectors_ranked = rank(optim_vectors)
            #     query_vectors_ranked = rank(query_vectors)
            #     optim_vectors_ranked_means = optim_vectors_ranked.mean(dim=1, keepdim=True)
            #     optim_vectors_ranked_centered = optim_vectors_ranked - optim_vectors_ranked_means
            #     optim_vectors_ranked_centered_normed = optim_vectors_ranked_centered / torch.norm(optim_vectors_ranked_centered, dim=1, keepdim=True).clamp(min=1e-7)
            #     query_vectors_ranked_means = query_vectors_ranked.mean(dim=1, keepdim=True)
            #     query_vectors_ranked_centered = query_vectors_ranked - query_vectors_ranked_means   
            #     query_vectors_ranked_centered_normed = query_vectors_ranked_centered / torch.norm(query_vectors_ranked_centered, dim=1, keepdim=True).clamp(min=1e-7)
            #     all_dists = torch.matmul(optim_vectors_ranked_centered_normed, query_vectors_ranked_centered_normed.t())
            #     all_dists.neg_()
            #     del optim_vectors_ranked, query_vectors_ranked
            # elif self.dist_measure == "l2":
            #     el_diffs = optim_vectors.unsqueeze(1) - query_vectors.unsqueeze(0) # optim_vecs, lib_vecs, dims
            #     all_dists = torch.sum(el_diffs ** 2, dim=-1) # optim_vecs, lib_vecs - note, maybe different distance measures
            # else:
            #     raise ValueError(f"Unknown dist measure: {self.dist_measure}")
            
            # if should_clean_query:
            #     del query_vectors
            
            # # depth_mask = [self.syntax.get_depth(t) + context.pos.at_depth <= self.syntax.max for t in query_terms]

            # best_term_vec_dists = torch.min(all_dists, dim=0).values # lib_vecs
            # del all_dists
            # sort_ids = torch.argsort(best_term_vec_dists)
            # sorted_dists = best_term_vec_dists[sort_ids]
            # ordered_terms = [query_terms[i] for i in sort_ids.tolist()]
            # context.lib_term_dists = sorted_dists.tolist()
            # context.lib_term_order = ordered_terms
            # del best_term_vec_dists, sort_ids, sorted_dists
            # # if len(ordered_terms) > self.num_lib_terms:
            # #     ordered_terms = ordered_terms[:self.num_lib_terms]
            # #     context.lib_term_dists = context.lib_term_dists[:self.num_lib_terms]
            # #     context.lib_term_order = ordered_terms

        best_dist = None #ordered_kb[0][0]

        if self.best_by_metric == "loss": # run all optimizations of ordered terms abd pick best 
            sz = min(self.num_lib_terms, len(ordered_kb))
            final_terms = []
            final_dists = []
            final_fillings = []
            for i in range(len(ordered_kb)):
                if len(final_terms) >= sz:
                    break
                dist, k, stripped_filling, b = ordered_kb[i]
                # if stripped_filling == context.stripped_pos_term:
                #     continue # do not replace by the same term (even stripped)
                filling = self.build_lincomb(k, stripped_filling, b)
                if isinstance(filling, Value) and isinstance(context.pos.term, Value) and torch.isclose(filling.value, context.pos.term.value, atol=self.identity_atol, rtol=self.identity_rtol):
                    continue                
                new_term = self.combine_new_term(context.term, context.pos, filling)
                if new_term is None:
                    continue
                if best_dist is None:
                    best_dist = dist
                self.evaluator.eval(new_term)
                if not self.semantics.is_valid(new_term):
                    continue
                if (self.pick_best_dists and dist > best_dist):
                    break
                final_terms.append(new_term)
                final_fillings.append(filling)
                final_dists.append(dist)
                self.num_total_fills += 1

            # context.lib_term_dists = final_dists
            # context.lib_term_order = final_fillings
            if len(final_terms) == 0:
                context.status = "no_valid_fill"
                self.finalize_context(context)
                return
            final_losses = self.fitness.get_fitness(final_terms, return_type="tensor")
            context.final_losses = final_losses.tolist()
            sort_ids = torch.argsort(final_losses)
            best_loss_id = sort_ids[0].item()
            context.final_loss = final_losses[best_loss_id].item()
            context.final_term = final_terms[best_loss_id]     
            context.filling = final_fillings[best_loss_id]   
            context.dist = final_dists[best_loss_id]

            conts = []
            for i in range(1, len(sort_ids)):
                term_id = sort_ids[i].item()
                filling = final_fillings[term_id]
                loss = final_losses[term_id].item()
                final_term = final_terms[term_id]
                dist = final_dists[term_id]
                cont = LossBasedContinuation(context, filling=filling, final_term=final_term, final_loss=loss, dist=dist)
                conts.append(cont)

            del final_losses, sort_ids            

            if len(conts) > 0:
                self.term_continuations.setdefault(context.term, []).extend(conts)

        elif self.best_by_metric == "dist":
            context.final_loss = float('inf')
            context.final_term = None    
            next_i = 0   
            ordered_kb = ordered_kb[:self.num_lib_terms * 2] #
            for (dist, k, stripped_filling, b) in ordered_kb:
                next_i += 1
                # if stripped_filling == context.stripped_pos_term:
                #     continue # do not replace by the same term (even stripped)
                filling = self.build_lincomb(k, stripped_filling, b)
                # assumes that lib has at least one constant
                if isinstance(filling, Value) and not isinstance(stripped_filling, Value):
                    continue 
                if isinstance(filling, Value) and isinstance(context.pos.term, Value) and torch.isclose(filling.value, context.pos.term.value, atol=self.identity_atol, rtol=self.identity_rtol):
                    continue
                context.filling = filling
                new_term = self.combine_new_term(context.term, context.pos, filling)
                if new_term is not None:
                    self.evaluator.eval(new_term)
                    if not self.semantics.is_valid(new_term):
                        continue
                    if best_dist is None:
                        best_dist = dist
                    loss = self.fitness.get_fitness(new_term).item()
                    context.final_loss = loss
                    context.dist = dist
                    context.final_term = new_term
                    break
            if context.final_term is not None:
                context.final_losses = [context.final_loss]

            if next_i > 0:
                if self.pick_best_dists:
                    conts = []
                    conts_size = self.num_lib_terms - 1
                    end_index = next_i
                    while end_index < len(ordered_kb) and len(conts) < conts_size:
                        cur_dist = ordered_kb[end_index][0]
                        if cur_dist > best_dist:
                            break
                        end_index += 1
                else: 
                    end_index = min(len(ordered_kb), next_i + self.num_lib_terms - 1)
                    # index_range = range(next_i, end_index)

                conts = [DistBasedContinuation(context, *ordered_kb[i]) for i in range(next_i, end_index)]
                # context.lib_term_order = [ordered_kb[i][2] for i in range(next_i - 1, end_index)]
                # context.lib_term_dists = [ordered_kb[i][0] for i in range(next_i - 1, end_index)]
                if len(conts) > 0:
                    self.term_continuations.setdefault(context.term, []).extend(conts)
        else:
            raise ValueError(f"Unknown best_by_metric: {self.best_by_metric}")


        self.finalize_context(context)

        return
    
    def mutate_position(self, _: Term, context: TermMutationContext) -> TermMutationContext | None:
        self.mutation_log.append(context)
        self.mutation_log_per_term.setdefault(context.term, []).append(context)
        self.mutate_one_position(context)
        if (context.status == "active") or (self.allow_no_better and context.status == "no_better"):
            assert context.final_term is not None
            return context
        return None 

    def population_order(self, population: Sequence[Term]) -> Generator[int, None, None]:
        ''' Assumes population is ordered by term complexity already. Override shuffle '''
        return range(len(population))      
       
    def select_positions(self, term: Term) -> Generator[TermMutationContext, None, None]:   

        if term not in self.term_contexts:
            if not self.semantics.is_valid(term) or isinstance(term, Value): # do not optimize invalid terms
                return
            cur_gen = self.get_cur_gen()
            positions = self.syntax.get_positions(term)
            priorities = self.position_strategy(term, positions) # should be cached if necessary
            start_loss = self.fitness.get_fitness(term).item()
            contexts = []            
            for priority, pos in zip(priorities, positions):
                _, stripped_pos_term, _ = self.decompose_lincomb(pos.term)
                context = TermMutationContext(cur_gen, term, pos, stripped_pos_term, 
                                              pos_priority=priority, start_loss=start_loss) 
                contexts.append(context)
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

            yield context

        backtrack_term = self.get_backtrack_term(term) # here only can go to lineage
        if backtrack_term is not None:
            yield from self.select_positions(backtrack_term)         

        return
    
    def mutate_term(self, term):
        base_fn = super().mutate_term
        def fn():
            res = base_fn(term)
            return res        
        # if self.debug:
        #     res, time = timed(fn)()
        #     print(f"{time}ms: {term}")
        # else:            
        res = fn()
        return res

    def add_to_lineage(self, parent: Term, child: TermMutationContext):
        if child.final_term is not None:
            super().add_to_lineage(parent, child.final_term)

    def get_backtrack_term(self, term: Term, backtrack: Literal["lineage", "worse_fillings"] | None = None) -> Term | None:
        if len(self.term_contexts.get(term, [])) > 0:
            return term # try self other position before any backtrack
        backtrack = backtrack or self.backtrack

        # first, try to refill term_contexts from cached continuations 

        term_continuations = self.term_continuations.pop(term, [])
        if len(term_continuations) > 0:
            term_continuations.sort(key=lambda c: c.get_priority()) # accross positions 
            for cont_id, cont in enumerate(term_continuations):
                if cont_id >= self.num_conts:
                    break
                self.add_context_continuation(cont, cont_id + 1)
            return term

        if backtrack is None:
            return None
        if backtrack == "worse_fillings": # first try to find the best no_better attempt in the log and reattempt it
            cur_num_worse_fillings = self.worse_filling_counts.setdefault(term, 0)
            if cur_num_worse_fillings < self.num_worse_fillings:
                self.worse_filling_counts[term] = cur_num_worse_fillings + 1
                logs = self.mutation_log_per_term.get(term, [])
                filtered_logs = [l for l in logs if l.status == "no_better" and not l.reattempt and math.isfinite(l.final_loss) and math.isfinite(l.dist[-1])] # assert final_term is set already + loss and dist
                best_among_worst = None
                for log in filtered_logs:
                    assert log.final_term is not None and log.final_loss is not None
                    if best_among_worst is None or log.final_loss < best_among_worst.final_loss:
                        best_among_worst = log
                if best_among_worst is not None:
                    best_among_worst.reattempt = True # to filter out on next reattempt 
                    return best_among_worst.final_term # NOTE: returns final_term itself as it is ready
            else:
                pass
        elif backtrack == "rand":
            depth = self.rnd.integers(1, self.backtrack_rand_grow_depth + 1)
            rand_term = self.syntax.grow(depth)
            self.evaluator.eval(rand_term)
            return rand_term
        # backtrack == "lineage" - pick prev parent
        cur_lineage = self.get_term_history(term)
        filtered_lineage = [fp for cur_terms in cur_lineage 
                                for fp in [[t for t in cur_terms 
                                                if t in self.term_contexts and len(self.term_contexts[t]) > 0]]
                                if len(fp) > 0]
        if len(filtered_lineage) > 0:           
            # backtrack_terms = filtered_lineage[0] # parent, or random
            # pick best parent to continue 
            backtrack_parents = [p for ps in filtered_lineage for p in ps]
            if len(backtrack_parents) == 1:
                return backtrack_parents[0]
            bp_fitness = self.fitness.get_fitness(backtrack_parents, return_type="tensor")
            best_bp_id = torch.argmin(bp_fitness).item()
            # best_bp_id = self.rnd.choice(len(backtrack_parents))
            del bp_fitness
            parent_term = backtrack_parents[best_bp_id]
            return parent_term
        # resort to random restart when whole lineage is exhausted
        depth = self.rnd.integers(1, self.backtrack_rand_grow_depth + 1)
        rand_term = self.syntax.grow(depth)
        self.evaluator.eval(rand_term)
        return rand_term
        # return None

    def __call__(self, population):   

        # if self.get_cur_gen() == 0: # first call - add lib to popualtion 
        #     population = self.lib_terms             
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
                if (context.start_loss > 1e-6) and (context.final_loss < context.start_loss):
                    loss_percent_left = context.final_loss / context.start_loss
                    self.stats_loss_improvement.append(loss_percent_left)
                    self.stats_dists.append(context.dist[-1])
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
                ctx_data = dict(context.__dict__)
                ctx_data.pop("stripped_pos_term", None)
                ctx_data.pop("reattempt", None)
                json.dump(ctx_data, self.log_file, default=metrics_serializer)
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
            new_children = list(set.difference(self.term_frontier, self.deadends))
            pass
        if len(new_children) == 0:
            raise EvSearchTermination("DEADEND")
        return new_children
    
    def get_finalizer(self):

        # compute final correlation between dist and loss change
        try:
            x = np.array(self.stats_dists)
            y = np.array(self.stats_loss_improvement)
            pearson_dist_loss, pearson_dist_loss_p_value = pearsonr(x, y)
            spearman_dist_loss, spearman_dist_loss_p_value = spearmanr(x, y)
        except Exception as e:
            pearson_dist_loss = None
            pearson_dist_loss_p_value = None
            spearman_dist_loss = None
            spearman_dist_loss_p_value = None

        self.add_metrics(
            num_better_fills=self.num_better_fills,
            num_total_fills=self.num_total_fills,
            num_holes_created=self.num_holes_created,
            num_terms_optimized=self.num_terms_optimized,
            tried_optim_terms=len(self.tried_optim_terms),
            tried_optim_terms_hit=self.tried_optim_terms_hit,
            pearson_dist_loss=pearson_dist_loss,
            pearson_dist_loss_p_value=pearson_dist_loss_p_value,
            spearman_dist_loss=spearman_dist_loss,
            spearman_dist_loss_p_value=spearman_dist_loss_p_value,
            # "num_terms_created": self.num_terms_created,
            # tabu_positions=sum(len(v) for v in self.tabu_positions.values())
            tabu_positions=len(self.tabu_set)
            )
        def finalizer():
            self.log_file and self.log_file.close()
        return finalizer
        