from pyomo.contrib.pynumero import mumps_available
from pyomo.contrib.pynumero.extensions.hsl import _MA27_LinearSolver
if _MA27_LinearSolver.available():
    from pyomo.contrib.pynumero.linalg.solvers.ma27_solver import MA27LinearSolver
if mumps_available:
    from pyomo.contrib.pynumero.linalg.solvers.mumps_solver import MUMPSSymLinearSolver
