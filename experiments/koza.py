''' Running GPSolver with classic Koza pipeline:
    RHH + TS(7) + RPM(10%) + RPX(90%)
'''

from t_search.operators import RHH, TS, RPM, RPX
from t_search import GPSolver
from .base import run_args

def get_solver(**kwargs):
    solver = GPSolver(
        init=RHH(),
        pipeline=[ TS(tournament_size=7), RPM(rate=0.1), RPX(rate=0.9) ],
        **kwargs
    )
    return solver

if __name__ == "__main__":
    run_args(get_solver)