from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver


class GurobiPersistent(PersistentSolver, GurobiDirect):
    pass
