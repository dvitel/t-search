
from typing import Callable, Optional

import numpy as np

from .validation import get_pos_constraints, is_valid

from .stats import get_depth
from .generation import Builders, TermGenContext
from .traverse import TRAVERSAL_EXIT, TRAVERSAL_EXIT_NODE, postorder_traversal
from .term import Term, TermPos


def enum_occurs(new_term: Term, some_occurs: dict, fn = lambda *_:()):

    def _enter_new_child(t, *_):
        cur_occur = some_occurs.setdefault(t, 0)        

    def _exit_new_child(t, _, p):
        res = fn(t, some_occurs[t], p)
        some_occurs[t] += 1
        return res

    postorder_traversal(new_term, _enter_new_child, _exit_new_child)

def replace_pos(pos: TermPos, with_term: Term, builders: Builders) -> Optional[Term]:
    if with_term is None:
        return None
    cur_pos = pos
    new_term = with_term
    while cur_pos.parent is not None:
        parent = cur_pos.parent.term
        term_i = cur_pos.pos
        args = parent.get_args()
        new_parent_term_args = tuple((*args[:term_i], new_term, *args[term_i + 1:]))   
        builder = builders.get_term_builder(parent)
        new_parent_term = builder.fn(*new_parent_term_args)
        if new_parent_term is None:
            return None
        new_term = new_parent_term
        cur_pos = cur_pos.parent

    return new_term

def replace_pos_protected(pos: TermPos, with_term: Term, builders: Builders,
                            depth_cache: dict[Term, int], counts_cache: dict[Term, np.ndarray],
                            pos_context_cache: dict[tuple[Term, int], TermGenContext],
                            max_term_depth: int = 17) -> Optional[Term]:

    if pos.at_depth + get_depth(with_term, depth_cache) > max_term_depth:
        return None 

    pos_context = get_pos_constraints(pos, builders, counts_cache, pos_context_cache)

    if not is_valid(with_term, builders=builders, 
                            counts_cache=counts_cache,
                            root_context=pos_context):
        return None

    new_term = replace_pos(pos, with_term, builders)

    return new_term

def replace_fn(root: Term,
            get_replacement_fn: Callable[[Term, int], Optional[Term]],
            builders: Builders) -> Optional[Term]:

    occurs = {}

    # replacement = {}
    arg_stack = [[]]

    def _replace_enter(term: Term, term_i: int, parent: Term):
        cur_occur = occurs.setdefault(term, 0)
        new_term = get_replacement_fn(term, cur_occur)
        if new_term is not None:
            if isinstance(new_term, Term):
                arg_stack[-1].append(new_term)
                enum_occurs(term, occurs)
                return TRAVERSAL_EXIT_NODE
            else:
                arg_stack.clear()
                return TRAVERSAL_EXIT
        else:
            arg_stack[-1].append(term)
        arg_stack.append([])

    def _replace_exit(term: Term, term_i: int, parent: Term):
        new_args = arg_stack.pop()
        builder = builders.get_term_builder(term)
        new_term = builder.fn(*new_args)
        if new_term is None:
            arg_stack.clear()
            return TRAVERSAL_EXIT
        arg_stack[-1][term_i] = new_term
        occurs[term] += 1

    postorder_traversal(root, _replace_enter, _replace_exit)

    return None if len(arg_stack) == 0 else arg_stack[-1][-1]
