''' Trivial targets: constants and variables. Should be checked first. '''
import math
import numpy as np
import pytest
from t_search import GPSolver

CONST_CASES = [0.0, math.pi, -math.e, 1e7]
@pytest.mark.parametrize("target", CONST_CASES, ids=[f"c={c}" for c in CONST_CASES])
def test_const(target: float):
    ''' Solving the target with trivial check '''
    atol = 1e-2
    solver = GPSolver(operators=[], max_gen=0, atol=atol) # pipeline does not matter as we test trivial cases
    n = 100
    x = list(range(n))
    free_vars = [x]
    target = [target + atol * (np.random.rand() - 0.5) for _ in range(n)]
    solver._reset_state(free_vars, target)
    assert solver._check_trivial(), "Trivial constant target was not solved"
    pass 

# CONST_CASES = [0.0, math.pi, -math.e]
# @pytest.mark.parametrize("target", CONST_CASES, ids=[f"c={c}" for c in CONST_CASES])
# def test_not_const(target: float):
#     ''' Solving the target with trivial check '''
#     atol = 1e-2
#     solver = GPSolver(pipeline=[], max_gen=0, atol=atol) # pipeline does not matter as we test trivial cases
#     n = 100
#     x = list(range(n))
#     free_vars = [x]
#     target = [target + atol * 10 * (np.random.rand() - 0.5) for _ in range(n)]
#     solver._reset_state(free_vars, target)
#     assert not solver._check_trivial(), "Big variation assumed constant"
#     pass 

def test_var():
    ''' Solving the target with trivial check '''
    num_vars: int = 5
    solver = GPSolver(operators=[], max_gen=0) # pipeline does not matter as we test trivial cases
    n = 100
    xs = [[float(i) - var_id for i in range(n)] for var_id in range(num_vars)]
    for target_id in range(num_vars):
        target = list(xs[target_id])
        solver._reset_state(xs, target)
        assert solver._check_trivial(), "Trivial variable target was not solved"
    pass


