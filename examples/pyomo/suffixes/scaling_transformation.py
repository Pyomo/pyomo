#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pe

### Create the example model
model = pe.ConcreteModel()
model.s = pe.Set(initialize=[1,2])
model.z = pe.Var(initialize=100.0, bounds=(-100, 100.0))
model.x = pe.Var(model.s, bounds=(-10.0,10.0), initialize={1: 1.0, 2: -10.0})
model.obj = pe.Objective(expr=model.x[1] + 2.0*model.x[2] + 3.0*model.z)

def ineq_rule(m, i):
    if i == 1:
        return m.z + m.x[1] >= -1.0
    else:
        return -m.z + m.x[1] <= 1.0
model.inequ = pe.Constraint(model.s, rule=ineq_rule)

model.eq = pe.Constraint(expr=model.x[2] == model.x[1]+2.0)


### Declare the scaling_factor suffix 
model.scaling_factor = pe.Suffix(direction=pe.Suffix.EXPORT)
# set objective scaling factor
model.scaling_factor[model.obj] = 1.0/281.0 # should give scaled objective value of 1.0
# set variable scaling factor
model.scaling_factor[model.z] = 100.0 # should have scaled initial value of 1.0
model.scaling_factor[model.x[2]] = -10.0 # should have scaled initial value of 1.0
# set constraint scaling factor
model.scaling_factor[model.inequ[1]] = 1.0/101.0 # should give a scaled value of 1.0 at initial pt
model.scaling_factor[model.inequ[2]] = -1.0/99.0 # should give a scaled value of 1.0 at initial pt
model.scaling_factor[model.eq] = 1.0/13.0 # should give a scaled value of 1.0 at initial pt

###
# scale and solve the problem
###
unscaled_model = model
# transform the unscaled model to a scaled model
scaling_tx = pe.TransformationFactory('core.scale_model')
scaled_model = scaling_tx.create_using(unscaled_model)
# print the scaled model
print('*** SCALED MODEL ***')
scaled_model.pprint()
# solve the scaled model
pe.SolverFactory('glpk').solve(unscaled_model)
# propagate the solution back to the unscaled model
scaling_tx.propagate_solution(scaled_model, unscaled_model)
# print the usncaled model with propagated solution
print('*** UNSCALED MODEL ***')
scaled_model.pprint()

compare_solutions = False
if compare_solutions:
    original_model = model.clone()
    unscaled_model = model.clone()
    pe.SolverFactory('glpk').solve(unscaled_model)

    print('*** UNSCALED MODEL SOLN ***')
    unscaled_model.display()

    scaling_tx = pe.TransformationFactory('core.scale_model')
    scaled_model = scaling_tx.create_using(unscaled_model)
    pe.SolverFactory('glpk').solve(scaled_model)
    scaling_tx.propagate_solution(scaled_model, original_model)
    print('*** SCALED MODEL SOLN --> BACK TO UNSCALED MODEL ***')
    original_model.display()


