

from t_search.syntax import parse_term


def test_terms():

    # tests
    t1, _ = parse_term("(f (f X (f x (f x)) (f x (f x))))")
    print(str(t1))
    t1_str1 = term_to_str(t1)
    t2, _ = parse_term("(f (f (f x x) Y Y))")
    t3, _ = parse_term("(f Z)")
    # b = UnifyBindings()
    # res = unify(b, points_are_equiv, t1, t2, t3)
    pass


    t1_str = "(f (f (f x x) (f 1.42 (f x)) (f 1.42 (f x))))"
    # t1_str = "(f x x 1.43 1.42)"
    t1, _ = parse_term(t1_str)

    depth = get_depth(t1)
    print(depth)
    pass    

    print(str(t1))
    assert str(t1) == t1_str, f"Expected {t1_str}, got {str(t1)}"
    pass
    # t1, _ = parse_term("(f x y x x x x x)")
    # t1, _ = parse_term("(inv (exp (mul x (cos (sin (exp (add 0.134 (exp (pow x x)))))))))")
    # t1, _ = parse_term("(pow (pow x0 1.81) (log 1.02))")
    p1, _ = parse_term("(f (f X X) Y Y)")
    # p1, _ = parse_term("(... (f 1.42 X))")
    # p1, _ = parse_term("(exp (... (exp (... (exp .)))))")
    # p1, _ = parse_term("(... pow (pow . .))")
    # p1, _ = parse_term("(... exp (exp X))")
    # p1, _ = parse_term("(f x X *)")

    p1_str = term_to_str(p1)
    term_cache = {}
    ut1 = unique_term(t1, term_cache)
    up1 = unique_term(p1, term_cache)
    matches = match_terms(ut1, up1, traversal="bottom_up", first_match=True)
    matches = [(str(m[0]), {k:str(v) for k, v in m[1].bindings.items()}) for m in matches]
    pass

    # res, _ = parse_term("  \n(   f   (g    x :0:1)  (h \nx) :0:12)  \n", 0)
    t1, _ = parse_term("  \n(   f   (g    x)  (h \nx))  \n", 0)
    leaves = collect_terms(t1, lambda t: isinstance(t, Variable))
    # bindings = bind_terms(leaves, 1)
    bindings = {parse_term("x")[0]: 1}
    print(str(t1))
    ev1 = evaluate(t1, { "f": lambda x, y: x + y, 
                         "g": lambda x: x * 2, 
                         "h": lambda x: x ** 2 }, lambda _,x: bindings.get(x), 
                   lambda _,*x: bindings.setdefault(*x))

    pass        