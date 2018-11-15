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

###
# create the original unscaled model
###
model = pe.ConcreteModel()
model.x = pe.Var([1,2,3], bounds=(-10,10), initialize=5.0)
model.z = pe.Var(bounds=(10,20))
model.obj = pe.Objective(expr=model.z + model.x[1])

# demonstrate scaling of duals as well
model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
model.rc = pe.Suffix(direction=pe.Suffix.IMPORT)
        
def con_rule(m, i):
    if i == 1:
        return m.x[1] + 2*m.x[2] + 1*m.x[3] == 4.0
    if i == 2:
        return m.x[1] + 2*m.x[2] + 2*m.x[3] == 5.0
    if i == 3:
        return m.x[1] + 3.0*m.x[2] + 1*m.x[3] == 5.0
model.con = pe.Constraint([1,2,3], rule=con_rule)
model.zcon = pe.Constraint(expr=model.z >= model.x[2])

###
# set the scaling parameters
###
model.scaling_factor = pe.Suffix(direction=pe.Suffix.EXPORT)
model.scaling_factor[model.obj] = 2.0
model.scaling_factor[model.x] = 0.5
model.scaling_factor[model.z] = -10.0
model.scaling_factor[model.con[1]] = 0.5 
model.scaling_factor[model.con[2]] = 2.0
model.scaling_factor[model.con[3]] = -5.0
model.scaling_factor[model.zcon] = -3.0

###
# build and solve the scaled model
###
scaled_model = pe.TransformationFactory('core.scale_model').create_using(model)
pe.SolverFactory('glpk').solve(scaled_model)


###
# propagate the solution back to the original model
###
pe.TransformationFactory('core.scale_model').propagate_solution(scaled_model, model)

# print the scaled model
scaled_model.pprint()

# print the solution on the original model after backmapping
model.pprint()

compare_solutions = True
if compare_solutions:
    # compare the solution of the original model with a clone of the
    # original that has a backmapped solution from the scaled model
    
    # solve the original (unscaled) model
    original_model = model.clone()
    pe.SolverFactory('glpk').solve(original_model)

    # create and solve the scaled model
    scaling_tx = pe.TransformationFactory('core.scale_model')
    scaled_model = scaling_tx.create_using(model)
    pe.SolverFactory('glpk').solve(scaled_model)

    # propagate the solution from the scaled model back to a clone of the original model
    backmapped_unscaled_model = model.clone()
    scaling_tx.propagate_solution(scaled_model, backmapped_unscaled_model)

    # compare the variable values
    print('\n\n')
    print('%s\t%12s           %18s' % ('Var', 'Orig.', 'Scaled -> Backmapped'))
    print('=====================================================')
    for v in original_model.component_data_objects(ctype=pe.Var, descend_into=True):
        cuid = pe.ComponentUID(v)
        bv = cuid.find_component_on(backmapped_unscaled_model)
        print('%s\t%.16f\t%.16f' % (v.local_name, pe.value(v), pe.value(bv)))
    print('=====================================================')



