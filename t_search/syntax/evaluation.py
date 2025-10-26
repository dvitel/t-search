
from typing import Any, Callable
from .term import Op, Term
from .traverse import TRAVERSAL_EXIT_NODE, postorder_traversal


def evaluate(root: Term, ops: dict[str, Callable],
                get_binding: Callable[[Term, Term], Any] = lambda ti: None,
                set_binding: Callable[[Term, Term, Any], Any] = lambda ti,v:()) -> Any:
    ''' Fully or partially evaluates term (concrete or abstract) '''
    
    args_stack = [[]]
    def _enter_args(term: Term, *_):
        res = get_binding(root, term)
        if res is not None:
            args_stack[-1].append(res)
            return TRAVERSAL_EXIT_NODE
        args_stack.append([])
        
    def _exit_term(term: Term, *_):
        args = args_stack.pop()
        res = None
        if isinstance(term, Op) and all(arg is not None for arg in args):
            op_fn = ops[term.op_id]
            res = op_fn(*args)
        if res is not None:            
            set_binding(root, term, res)
        # else:
        #     pass
        args_stack[-1].append(res)

    postorder_traversal(root, _enter_args, _exit_term)

    return args_stack[0][0]    
