from pyomo.contrib.pynumero import mumps_available, ma27_available
if ma27_available:
    from pyomo.contrib.pynumero.linalg.solvers.ma27_solver import MA27LinearSolver
if mumps_available:
    from pyomo.contrib.pynumero.linalg.solvers.mumps_solver import MUMPSSymLinearSolver
