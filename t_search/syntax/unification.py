
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

from .traverse import TRAVERSAL_EXIT, postorder_traversal

from .term import AnyOneWildard, MetaVariable, Op, RepeatWildcard, Term, Variable, is_ellipsis


@dataclass(eq=False, unsafe_hash=False)
class UnifyBindings:
    bindings: dict[str, Term] = field(default_factory=dict)
    renames: dict[str, str] = field(default_factory=dict)

    def copy(self) -> 'UnifyBindings':
        res = UnifyBindings()
        res.bindings = self.bindings.copy()
        res.renames = self.renames.copy()
        return res
    
    def update_with(self, other: 'UnifyBindings'):
        self.bindings.update(other.bindings)
        self.renames.update(other.renames)

    def get(self, *keys) -> tuple[Term, ...]:
        res = tuple(self.bindings.get(self.renames.get(k, k), None) for k in keys)
        return res
    
    def set(self, key: str, value: Term):
        self.bindings[key] = value

    def set_same(self, keys: list[str], to_key: str):
        to_key = self.renames.get(to_key, to_key)
        for k in keys:
            if k != to_key and k not in self.renames:
                self.renames[k] = to_key
    
def _points_are_equiv(ts: Sequence[Term], args: Sequence[Sequence[Term]]) -> bool:
    # arg_counts = [(len(sf), len(s) > 0 and takes_many_args(s[-1]))
    #               for t in ts 
    #               for s in [t.get_args()] 
    #               for sf in [rstrip(s)]]
    # max_count = max(ac for ac, _ in arg_counts)
    first_term = ts[0]
    first_args = args[0]
    def are_same(term1: Term, term2: Term) -> bool:
        if type(term1) != type(term2):
            return False
        if isinstance(term1, Op):
            if term1.op_id != term2.op_id:
                return False
            return True 
        return term1 == term2  # assuming impl of _eq or ref eq     
    res = all(are_same(t, first_term) and \
              (len(a) == len(first_args))
              for t, a in zip(ts, args))
    return res

def set_prev_match(prev_matches: dict[tuple, UnifyBindings | None], 
                   b: UnifyBindings, terms: tuple[Term, ...], match: bool) -> bool:
    prev_matches[terms] = b.copy() if match else None
    return match 

def unify(b: UnifyBindings, *terms: Term,
            prev_matches: dict[tuple, UnifyBindings | None]) -> bool:
    ''' Unification of terms. Uppercase leaves are meta-variables, 

        Note: we do not check here that bound meta-variables recursivelly resolve to concrete terms.
        This should be done by the caller.
    '''
    # if len(terms) == 2 and \
    #     (terms[0].arity() > 0) and (terms[1].arity() > 0) and \
    #     terms[0].op_id == terms[1]/op: # UnderWildcard check 
    #     args1 = terms[0].get_args()
    #     args2 = terms[1].get_args()
    #     if len(args1) > 1 and args1[0] == UnderWildcard:
    #         new_term = terms[1]
    #         new_pat = args[]

    if terms in prev_matches:
        m = prev_matches[terms]
        if m is not None:
            b.update_with(m)
            return True
        return False
    filtered_terms = [t for t in terms if t != AnyOneWildard]    
    if len(filtered_terms) < 2:
        return set_prev_match(prev_matches, b, terms, True)
    if any(t == RepeatWildcard for t in terms):
        return set_prev_match(prev_matches, b, terms, False)
    if len(filtered_terms) == 2:
        el_i = next((i for i, t in enumerate(filtered_terms) if is_ellipsis(t)), -1)
        if el_i >= 0:
            el_term = filtered_terms[el_i]
            if len(el_term.args) == 0:
                return set_prev_match(prev_matches, b, terms, False)
            other_term = filtered_terms[1 - el_i]
            new_pattern = el_term.args[-1]
            if len(el_term.args) > 1:
                name_var = el_term.args[0]
                if not(isinstance(name_var, Variable) and \
                    isinstance(other_term, Op) and \
                    (other_term.op_id == name_var.var_id)):
                    return set_prev_match(prev_matches, b, terms, False)
                matches = []
                for arg in other_term.get_args():
                    matches = match_terms(arg, new_pattern,
                                        with_bindings=b, first_match=True, 
                                        traversal="top_down",
                                        prev_matches=prev_matches)
                    if len(matches) > 0:
                        break 
                return set_prev_match(prev_matches, b, terms, len(matches) > 0)
            else:
                matches = match_terms(other_term, new_pattern, 
                                    with_bindings=b, first_match=True, 
                                    traversal="top_down",
                                    prev_matches=prev_matches)
                return set_prev_match(prev_matches, b, terms, len(matches) > 0)
    t_is_meta = [isinstance(t, MetaVariable) for t in filtered_terms]
    meta_operators = set([t.name for t, is_meta in zip(filtered_terms, t_is_meta) if is_meta])
    meta_terms = b.get(*meta_operators)
    bound_meta_terms = [bx for bx in meta_terms if bx is not None]
    concrete_terms = [t for t, is_meta in zip(filtered_terms, t_is_meta) if not is_meta]
    all_concrete_terms = bound_meta_terms + concrete_terms

    # expanding * wildcards
    all_concrete_terms_args = [t.get_args() for t in all_concrete_terms]
    max_len = max(len(args) for args in all_concrete_terms_args)
    first_repeats = [next((i for i, a in enumerate(args) if a == RepeatWildcard), -1)
                      for args in all_concrete_terms_args]
    expanded_args = [args if ri <= 0 else (args[:ri-1] + (args[ri-1],) * (max_len - len(args) + 2) + args[ri+1:])
                     for args, ri in zip(all_concrete_terms_args, first_repeats)]
    
    expanded_args = [[a for a in args if a != RepeatWildcard] for args in expanded_args]
    
    if len(all_concrete_terms) > 1:
        if not _points_are_equiv(all_concrete_terms, expanded_args):
            return set_prev_match(prev_matches, b, terms, False)
    unbound_meta_operators = [op for op, bx in zip(meta_operators, meta_terms) if bx is None]
    bound_meta_operators = [op for op, bx in zip(meta_operators, meta_terms) if bx is not None]
    if len(unbound_meta_operators) > 0:
        if len(bound_meta_operators) > 0:
            to_key = bound_meta_operators[0]
            b.set_same(unbound_meta_operators, to_key)
        else:
            to_key = unbound_meta_operators[0]
            if len(all_concrete_terms) > 0:
                term = all_concrete_terms[0]
                b.set(to_key, term)
            b.set_same(unbound_meta_operators, to_key)
    if len(all_concrete_terms) >= 2:
        for arg_tuple in zip(*expanded_args):
            if not unify(b, *arg_tuple, prev_matches=prev_matches):
                return set_prev_match(prev_matches, b, terms, False)
    return set_prev_match(prev_matches, b, terms, True)

MatchTraversal = Literal["bottom_up", "top_down"]

def match_terms(root: Term, pattern: Term,
                prev_matches: Optional[dict[tuple, UnifyBindings]] = None,
                with_bindings: UnifyBindings | None = None,
                first_match: bool = False,
                traversal: MatchTraversal = "bottom_up") -> list[tuple[Term, UnifyBindings]]:
    ''' Search for all occurances of pattern in term. 
        * is wildcard leaf. X, Y, Z are meta-variables for non-linear matrching
    '''
    if prev_matches is None:
        prev_matches = {}
    eq_terms = []
    def _match_node(t: Term, *_):
        # if exclude_root and t == root:
        #     return
        if with_bindings is not None:
            bindings = with_bindings.copy()
        else:
            bindings = UnifyBindings()
        if unify(bindings, t, pattern, prev_matches = prev_matches):
            eq_terms.append((t, bindings))
            if first_match:
                if with_bindings is not None:
                    with_bindings.update_with(bindings)
                return TRAVERSAL_EXIT
        pass
    if traversal == "top_down":
        postorder_traversal(root, _match_node, lambda *_: ())
    elif traversal == "bottom_up":
        postorder_traversal(root, lambda *_: (), _match_node)
    else:
        raise ValueError(f"Unknown match traversal: {traversal}")
    return eq_terms

def match_root(root: Term, pattern: Term,
                prev_matches: Optional[dict[tuple, UnifyBindings]] = None) -> Optional[UnifyBindings]:
    ''' Matches root only
    '''
    if prev_matches is None:
        prev_matches = {}
    bindings = UnifyBindings()
    if unify(bindings, root, pattern, prev_matches = prev_matches):
        return bindings
    return None
