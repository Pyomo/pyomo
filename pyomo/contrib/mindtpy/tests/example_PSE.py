from pyomo.environ import SolverFactory
import time
from pyomo.contrib.mindtpy.tests.flay03m import *
# from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
# model = EightProcessFlowsheet()
# with SolverFactory('mindtpy') as opt:
with SolverFactory('bonmin') as opt:
            print('\n Solving problem with Outer Approximation')
            start = time.time()
            # opt.solve(model, strategy='OA', init_strategy = 'rNLP')
            opt.solve(model)
            model.pprint()
            print(time.time()-start)