import time

from pyomo.environ import *
from pyomo.opt import TerminationCondition
from pyomo.pysp.annotations import (StochasticDataAnnotation,
                                    StageCostAnnotation,
                                    VariableStageAnnotation)
from pyomo.pysp.embeddedsp import (EmbeddedSP,
                                   UniformDistribution)
from pyomo.pysp.solvers import SPSolverFactory

# make this example deterministic
import random
random.seed(2352352342)

# This examples does the following:
# (1) Defines an EmbeddedSP by annotating a reference
#     model with information about variables stages
#     and stochastic parameters.
# (2) Obtains a solution using the deterministic
#     approximation, where stochastic parameters
#     are set to the expected value of their
#     distribution.
# (3) Generates an "external" SP by sampling from
#     the EmbeddedSP and uses DDSIP solver interface.
#     The DDSIP solver requires explicit annotations
#     that describe the locations of the stochastic data.
#     These annotations are automatically added when
#     the SP is created from an EmbeddedSP object.
# (4) The EmbeddedSP object is then re-sampled to
#     creating a test set that is used to evaluate
#     the solutions from steps (2) and (3).

d_dist = UniformDistribution(0, 100)
train_N = 150
test_N = 150
cplex = SolverFactory("cplex")
saa_solver = SPSolverFactory("ddsip") # "ef"

#
# Define the reference model
#
model = ConcreteModel()
model.x = Var(domain=Integers)
model.y = Var(domain=Integers)
model.t = Var(domain=Integers)
model.c = 1.0
model.b = 1.5
model.h = 0.1
model.d = Param(mutable=True)
model.cost = Expression([1,2], initialize=0.0)
model.cost[1].expr = 0
model.cost[2].expr = model.t
model.obj = Objective(expr= sum_product(model.cost))
model.cons = ConstraintList()
model.cons.add(model.t >= (model.c-model.b)*model.x + model.b*model.d)
model.cons.add(model.t >= (model.c+model.h)*model.x - model.h*model.d)
model.cons.add(model.y == -model.x)

#
# Embed an SP in the model using annotations
#
model.varstage = VariableStageAnnotation()
model.stagecost = StageCostAnnotation()
model.stochdata = StochasticDataAnnotation()
model.varstage.declare(model.x, 1)
model.stagecost.declare(model.cost[1], 1)
model.stagecost.declare(model.cost[2], 2)
model.stochdata.declare(model.d,
                        distribution=d_dist)
sp = EmbeddedSP(model)

#
# Solve the deterministic approximation
#
sp.set_expected_value()
status = cplex.solve(sp.reference_model)
assert status.solver.termination_condition == \
    TerminationCondition.optimal
ev_solution = sp.reference_model.x()

#
# Setup an explicit training sample and solve
#
train_sp = sp.generate_sample_sp(train_N)
results = saa_solver.solve(train_sp,
                           output_solver_log=True)
print(results)
saa_solution = list(results.xhat['root'].values())[0]

print("Deterministic Solution:  %s" % (ev_solution))
print("SAA Solution: %s" % (saa_solution))

#
# Evaluate the solutions on a test sample
#
ev_objectives = []
saa_objectives = []
for i in range(test_N):
    # samples into the reference model by default
    sp.sample()

    # evaluate the solution from the deterministic approximiation
    sp.reference_model.x.fix(ev_solution)
    status = cplex.solve(sp.reference_model)
    assert status.solver.termination_condition == \
        TerminationCondition.optimal
    ev_objectives.append(sp.reference_model.obj())

    # evaluate the solution from the SAA
    sp.reference_model.x.fix(saa_solution)
    status = cplex.solve(sp.reference_model)
    assert status.solver.termination_condition == \
        TerminationCondition.optimal
    saa_objectives.append(sp.reference_model.obj())

print("Average Test Objective for Deterministic Solution: %s"
      % (sum(ev_objectives)/float(test_N)))
print("Average Test Objective for SAA Solution: %s"
      % (sum(saa_objectives)/float(test_N)))
