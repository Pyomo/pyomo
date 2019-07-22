from pyomo.contrib.pynumero.linalg.solvers.mumps_solver import MUMPSSymLinearSolver
from pyomo.contrib.pynumero.extensions.hsl import _MA27_LinearSolver
if not _MA27_LinearSolver.available():
    from pyomo.contrib.pynumero.linalg.solvers.ma27_solver import MA27LinearSolver
