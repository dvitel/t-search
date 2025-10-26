from typing import Optional
import numpy as np

from .traverse import TRAVERSAL_EXIT, TRAVERSAL_EXIT_NODE, postorder_traversal
from .generation import Builders, TermGenContext
from .term import Term, TermPos

def get_counts(root: Term, builders: Builders, counts_cache: dict[Term, np.ndarray]) -> np.ndarray:

    counts_stack = [[]]

    def _enter_args(t: Term, *_):
        if t in counts_cache:
            counts_stack[-1].append(counts_cache[t])
            return TRAVERSAL_EXIT_NODE
        elif t.arity() == 0: # leaf
            builder = builders.get_term_builder(t)
            counts = builders.one_hot[builder.id]
            counts_stack[-1].append(counts)
            return TRAVERSAL_EXIT_NODE
        else:
            counts_stack.append([])

    def _exit_term(t: Term, *_):
        args = counts_stack.pop()
        counts = sum(args)
        builder = builders.get_term_builder(t)
        counts[builder.id] += 1
        counts_cache[t] = counts
        counts_stack[-1].append(counts)

    postorder_traversal(root, _enter_args, _exit_term)

    return counts_stack[-1][-1]

def get_pos_constraints(pos: TermPos, builders: Builders, counts_cache: dict[Term, np.ndarray],
                            pos_context_cache: dict[tuple[Term, int], TermGenContext]) -> TermGenContext:
    ''' Assumes that root was generated under correct constraints'''

    if (pos.term, pos.occur) in pos_context_cache:
        return pos_context_cache[(pos.term, pos.occur)]

    chain_to_root = [pos]
    while chain_to_root[-1].parent is not None:
        chain_to_root.append(chain_to_root[-1].parent)
    
    start_i = None
    for parent_i in range(1, len(chain_to_root)):
        parent = chain_to_root[parent_i]
        if (parent.term, parent.occur) in pos_context_cache:
            start_i = parent_i
            break

    if start_i is None:
        start_i = len(chain_to_root) - 1
        _context = builders.default_gen_context
    else:
        parent = chain_to_root[start_i]
        _context = pos_context_cache[(parent.term, parent.occur)]

    for parent_i in range(start_i, -1, -1):
        parent = chain_to_root[parent_i]
        parent_context = _context

        pos_context_cache[(parent.term, parent.occur)] = parent_context

        parent_builder = builders.get_term_builder(parent.term)

        # parent_counts = get_counts(parent.term, builders, counts_cache)
        # parent_one_hot = np.zeros(len(builders), dtype=np.int8)
        # parent_one_hot[parent_builder.id] = 1

        # assert parent_context.sat_args(parent_one_hot)
        # assert parent_context.sat(parent_counts)

        if parent_i == 0: 
            break
        
        arg = chain_to_root[parent_i - 1]
        arg_min_counts = parent_context.min_counts.copy()
        arg_max_counts = parent_context.max_counts.copy()
        arg_min_counts[parent_builder.id] -= 1
        arg_max_counts[parent_builder.id] -= 1
        if parent_builder.context_limits is not None:
            arg_max_counts = np.minimum(arg_max_counts, parent_builder.context_limits)
        arg_context = TermGenContext(
            min_counts=arg_min_counts,
            max_counts= arg_max_counts,
            arg_limits=parent_builder.arg_limits)
        if parent.term.arity() == 1:
            arg_context.min_counts[arg_context.min_counts < 0] = 0
            arg_context.max_counts[arg_context.max_counts < 0] = 0
            pos_context_cache[(arg.term, arg.occur)] = arg_context
            _context = arg_context
            continue

        child_counts = [ get_counts(child, builders, counts_cache) for child in parent.term.get_args() ]        
        other_counts = sum(cnts for child_i, cnts in enumerate(child_counts) if child_i != arg.pos)

        arg_context.min_counts -= other_counts
        arg_context.min_counts[arg_context.min_counts < 0] = 0
        arg_context.max_counts -= other_counts
        arg_context.max_counts[arg_context.max_counts < 0] = 0

        assert np.all(child_counts[arg.pos] <= arg_context.max_counts)
        assert np.all(child_counts[arg.pos] >= arg_context.min_counts)

        if arg_context.arg_limits is not None:
            arg_builder = builders.get_term_builder(arg.term)
            assert arg_context.arg_limits[arg_builder.id] > 0

        _context = arg_context

    res_context = pos_context_cache[(pos.term, pos.occur)]  

    return res_context

def is_valid(root: Term, *, builders: Builders, counts_cache: dict[Term, np.ndarray],                        
                        root_context: TermGenContext | None = None) -> bool:
    
    if root_context is None:
        root_context = builders.default_gen_context
                
    if root_context.arg_limits is not None:
        child_count = builders.zero.copy()
        builder = builders.get_term_builder(root)
        child_count[builder.id] += 1
        
        if np.any(child_count > root_context.arg_limits):
            return False
        
    counts = get_counts(root, builders, counts_cache)
    if np.any(counts > root_context.max_counts) or np.any(counts < root_context.min_counts):
        return False

    return True

def validate_term_tree(root: Term, *, builders: Builders, counts_cache: dict[Term, np.ndarray],                        
                        occurs: dict[Term, int] | None = None,
                        context_cache: dict[tuple[Term, int], TermGenContext] | None = None,
                        start_context: TermGenContext | None = None) -> Optional[Term]:
    
    if start_context is None:
        start_context = builders.default_gen_context

    if occurs is None:
        occurs = {}
    else:
        occurs = occurs.copy()

    pos_context_cache = {}

    is_valid = True 

    current_context_stack = [[start_context]]
    child_stack = [[root]]

    def _validate_enter(term: Term, term_i: int, *_):
        nonlocal is_valid        
        cur_occur = occurs.setdefault(term, 0)
        cur_context = current_context_stack[-1][term_i]
                
        if cur_context.arg_limits is not None:
            children = child_stack[-1]
            child_count = builders.zero.copy()
            for arg in children:
                builder = builders.get_term_builder(arg)
                child_count[builder.id] += 1
            
            if np.any(child_count > cur_context.arg_limits):
                is_valid = False
                return TRAVERSAL_EXIT
        

        counts = get_counts(term, builders, counts_cache)
        if np.any(counts > cur_context.max_counts) or np.any(counts < cur_context.min_counts):
            is_valid = False
            return TRAVERSAL_EXIT
        
        pos_context_cache[(term, cur_occur)] = cur_context

        if term.arity() == 0: # leaf
            occurs[term] += 1
            return TRAVERSAL_EXIT_NODE

        term_builder = builders.get_term_builder(term)

        arg_min_counts = cur_context.min_counts.copy()
        arg_max_counts = cur_context.max_counts.copy()
        arg_min_counts[term_builder.id] -= 1
        arg_max_counts[term_builder.id] -= 1
        if term_builder.context_limits is not None:
            arg_max_counts = np.minimum(arg_max_counts, term_builder.context_limits)

        child_stack.append(term.get_args())

        if term.arity() == 1:
            arg_context = TermGenContext(
                min_counts=arg_min_counts,
                max_counts=arg_max_counts,
                arg_limits=term_builder.arg_limits)
            arg_context.min_counts[arg_context.min_counts < 0] = 0
            arg_context.max_counts[arg_context.max_counts < 0] = 0
            current_context_stack.append([arg_context])
            return 
                    
        child_counts = [ get_counts(child, builders, counts_cache) for child in term.get_args() ]
        new_children = []
        term_arity = term.arity()
        for i in range(term_arity):
            child_context = TermGenContext(
                min_counts=(arg_min_counts if (i == term_arity - 1) else arg_min_counts.copy()),
                max_counts=(arg_max_counts if (i == term_arity - 1) else arg_max_counts.copy()),
                arg_limits=term_builder.arg_limits)

            other_counts = sum(cnts for child_i, cnts in enumerate(child_counts) if child_i != i)

            child_context.min_counts -= other_counts
            child_context.min_counts[child_context.min_counts < 0] = 0
            child_context.max_counts -= other_counts
            child_context.max_counts[child_context.max_counts < 0] = 0
            
            new_children.append(child_context)
            
        current_context_stack.append(new_children)

        pass 

    def _validate_exit(term: Term, *_):        
        current_context_stack.pop() 
        child_stack.pop()
        occurs[term] += 1

    postorder_traversal(root, _validate_enter, _validate_exit)

    if is_valid:
        if context_cache is not None:
            context_cache.update(pos_context_cache)
        return root
    return None

def get_pos_sibling_counts(position: TermPos, builders: Builders) -> np.ndarray:
    if position.sibling_count is None:        
        if position.parent is None or position.parent.term.arity() == 1:
            arg_counts = builders.zero.copy()
        else:
            arg_counts = get_immediate_counts(position.parent.term, builders)
            position_builder = builders.get_term_builder(position.term)
            arg_counts[position_builder.id] -= 1
        position.sibling_count = arg_counts
    return position.sibling_count

# def get_parent_path_counts(position: TermPos, builders: Builders) -> np.ndarray:
#     res = builders.zero.copy()
#     cur_pos = position.parent
#     while cur_pos is not None:
#         cur_builder = builders.get_term_builder(cur_pos.term)
#         res[cur_builder.id] += 1
#         cur_pos = cur_pos.parent
#     return res

def get_immediate_counts(root: Term, builders: Builders) -> np.ndarray:

    counts = builders.zero.copy()
    for arg in root.get_args():
        builder = builders.get_term_builder(arg)
        counts[builder.id] += 1

    return counts