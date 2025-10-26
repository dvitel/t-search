''' Base for many Term processing '''

from collections import deque
from typing import Any, Callable
from .term import Term


TRAVERSAL_EXIT_NODE = 1 
TRAVERSAL_EXIT = 2
    
def postorder_traversal(term: Term, enter_args: Callable, exit_term: Callable):
    ''' enter_args and exit_term are called with entered term and its parent.
        if enter_args returns True, args will be skipped, 
        if exit_term returns True, traversal will be terminated.
    '''
    q = deque([(0, term, 0, None)])
    while len(q) > 0:
        cur_arg_i, cur_term, cur_term_i, cur_parent = q.popleft()
        if cur_arg_i == 0:
            status = enter_args(cur_term, cur_term_i, cur_parent)
            if status == TRAVERSAL_EXIT:
                return
            if status == TRAVERSAL_EXIT_NODE:
                # should_end_traversal = exit_term(cur_term, cur_term_i, cur_parent)
                # if should_end_traversal:
                #     return
                continue
        cur_arg = cur_term.get_arg(cur_arg_i)
        if cur_arg is None:
            status = exit_term(cur_term, cur_term_i, cur_parent)
            if status == TRAVERSAL_EXIT:
                return
        else:            
            q.appendleft((cur_arg_i + 1, cur_term, cur_term_i, cur_parent))
            q.appendleft((0, cur_arg, cur_arg_i, cur_term))


def postorder_map(term: Term, fn: Callable, with_cache = False,
                    none_terminate = False) -> Any:  
    args_stack = [[]]
    term_cache = {}
    if with_cache:
        def add_res(t: Term, res: Any):
            term_cache[t] = res
    else:
        def add_res(*_):
            pass
    def _enter_args(t: Term, *_):
        if t in term_cache:
            processed_t = term_cache[t]
            args_stack[-1].append(processed_t)
            return TRAVERSAL_EXIT_NODE
        args_stack.append([])
    def _exit_term(t: Term, *_):
        term_processed_args = args_stack.pop()
        processed_t = fn(t, term_processed_args)
        add_res(t, processed_t)
        args_stack[-1].append(processed_t) #add to parent args
        if processed_t is None and none_terminate:
            return TRAVERSAL_EXIT
    postorder_traversal(term, _enter_args, _exit_term)
    return args_stack[-1][-1]

def collect_terms(root: Term, pred: Callable[[Term], bool]) -> list[Term]:
    ''' Find all leaves in root that are equal to term by name '''
    found = []
    def _exit_term(term: Term, *_):
        if pred(term):
            found.append(term)
    postorder_traversal(root, lambda *_: (), _exit_term)
    return found