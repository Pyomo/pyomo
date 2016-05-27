from pyomo.core import *
from pyomo.opt import SolverFactory, SolverManagerFactory

from DiseaseEstimation import model

# create the instance
instance = model.create('DiseaseEstimation.dat')

# define the solver and its options
solver = 'ipopt'
opt = SolverFactory( solver )
if opt is None:
    raise ValueError, "Problem constructing solver `"+str(solver)
opt.set_options('max_iter=2')

# create the solver manager
solver_manager = SolverManagerFactory( 'serial' )

# solve
results = solver_manager.solve(instance, opt=opt, tee=True, timelimit=None)
instance.load(results)

# display results
display(instance)
