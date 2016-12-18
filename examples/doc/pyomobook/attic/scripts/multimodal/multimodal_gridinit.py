from pyomo.environ import *
from pyomo.opt import SolverFactory
from collections import namedtuple
from math import pi
import numpy

model = ConcreteModel()

model.x = Var(bounds=(0,4))
model.y = Var(bounds=(0,4))

model.obj = Objective(expr= \
    (2 - cos(pi*model.x) - cos(pi*model.y)) * \
    (model.x**2) * (model.y**2))

# store attributes of a solution
Solution = namedtuple('Solution',
                      ['xinit','yinit',
                       'xsol','ysol',
                       'objective'])

# define a new class keep track of
# all unique solutions
class SolutionPool(object):

    def __init__(self, tol=1e-6):
        self.tol = tol
        self.unique_solns = []

    def update(self, xinit, yinit, model):

        candidate_soln = Solution(xinit, yinit,
                                  model.x.value, model.y.value,
                                  model.obj())

        # loop through the current list of unique solutions
        # and see if the candidate solution is new
        for soln in self.unique_solns:
            if not self.is_different(soln, candidate_soln):
                # candidate_soln is already in the list
                break
        else:
            # this gets executed with the for-loop
            # does not break
            self.unique_solns.append(candidate_soln)

    def is_different(self, soln1, soln2):
        if (abs(soln1.xsol - soln2.xsol) > self.tol) or \
           (abs(soln1.ysol - soln2.ysol) > self.tol):
            return True
        return False

    def pprint(self):
        print("SolutionPool (size=%s):"
              % (len(self.unique_solns)))
        for i, soln in enumerate(self.unique_solns, 1):
            print(" - solution%d" % (i))
            print("\tx0=%f, y0=%f" % (soln.xinit, soln.yinit))
            print("\tx*=%f, y*=%f" % (soln.xsol, soln.ysol))
            print("\tobjective=%f" % (soln.objective))

# setup the grid of starting points
xgrid = numpy.arange(2, 4.25, 0.25)
ygrid = numpy.linspace(2, 4, 8)

# loop through all the starting points and add
# solutions to the solution pool
solution_pool = SolutionPool(tol=1e-3)
cnt = 0
for xinit in xgrid:
    for yinit in ygrid:
        # initialize at the current grid point
        # and solve the problem
        model.x = xinit
        model.y = yinit

        with SolverFactory("ipopt") as solver:
            solver.solve(model)

        solution_pool.update(xinit, yinit, model)
        cnt += 1

    # print progress
    print("Percent Complete: %s%%"
          % (round(float(cnt)/(len(xgrid)*len(ygrid))*100)))

# print out the unique solutions
solution_pool.pprint()
