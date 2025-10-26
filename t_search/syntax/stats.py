
from typing import Generator, Optional

import numpy as np

from .traverse import TRAVERSAL_EXIT_NODE, postorder_traversal
from .term import Term, TermPos


def get_depth(term: Term, depth_cache: Optional[dict[Term, int]] = None) -> int:
    
    if depth_cache is None:
        depth_cache = {}
    
    def _enter_args(term: Term, *_):
        if (term in depth_cache) or (term.arity() == 0):
            return TRAVERSAL_EXIT_NODE

    def _exit_term(term: Term, *_):
        depth_cache[term] = 1 + max(depth_cache.get(a, 0) for a in term.get_args())

    postorder_traversal(term, _enter_args, _exit_term) 

    return depth_cache.get(term, 0)

def get_size(term: Term, size_cache: Optional[dict[Term, int]] = None) -> int:

    if size_cache is None:
        size_cache = {}

    def _enter_args(term: Term, *_):
        if (term in size_cache) or (term.arity() == 0):
            return TRAVERSAL_EXIT_NODE 

    def _exit_term(term: Term, *_):
        size_cache[term] = 1 + sum(size_cache.get(a, 1) for a in term.get_args())

    postorder_traversal(term, _enter_args, _exit_term)

    return size_cache.get(term, 1)

def get_positions(root: Term, pos_cache: dict[Term, list[TermPos]],
                    without_root: bool = True) -> list[TermPos]:
    ''' Bottom-up left-to-right collection of term positions, excluding root  '''

    if root in pos_cache:
        return pos_cache[root]

    positions: list[TermPos] = []
    at_depth = 0

    # arg_stack = [(TermPos(None), [])]
    parent_positions = [None]

    occurs = {}
    def _enter_args(term: Term, term_i, *_):
        nonlocal at_depth
        cur_occur = occurs.setdefault(term, 0)
        parent_pos = parent_positions[-1]
        term_pos = TermPos(term, cur_occur, term_i, at_depth,
                           parent = parent_pos)
        parent_positions.append(term_pos)
        at_depth += 1

    def _exit_term(*_):
        nonlocal at_depth
        at_depth -= 1
        cur_pos = parent_positions.pop()
        # if len(children_pos) > 0:
        #     cur_pos.depth = max(child.depth for child in children_pos) + 1
        # cur_pos.size += sum(child.size for child in children_pos)
        positions.append(cur_pos)
        occurs[cur_pos.term] += 1

    postorder_traversal(root, _enter_args, _exit_term)

    if without_root:
        positions.pop() # remove root 
    pos_cache[root] = positions

    return positions # last one is the root

def get_inner_terms(root: Term) -> list[Term]:
    present_terms = set()
    inner_terms = []
    
    def _inner_args(term: Term, *_):
        if term not in present_terms:
            inner_terms.append(term)
            present_terms.add(term)
    
    postorder_traversal(root, _inner_args, lambda *_: ()) 
    return inner_terms