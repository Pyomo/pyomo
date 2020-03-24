#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *
from pyomo.opt import SolverFactory

from six import itervalues

#import yaml

opt = SolverFactory('cplexamp',solve_io='nl')

kwds = {'pw_constr_type':'UB','pw_repn':'DCC','sense':maximize,'force_pw':True}

problem_names = []
problem_names.append("piecewise_multi_vararray")
problem_names.append("concave_multi_vararray1")
problem_names.append("concave_multi_vararray2")
problem_names.append("convex_multi_vararray1")
problem_names.append("convex_multi_vararray2")
problem_names.append("convex_vararray")
problem_names.append("concave_vararray")
problem_names.append("convex_var")
problem_names.append("concave_var")
problem_names.append("piecewise_var")
problem_names.append("piecewise_vararray")
problem_names.append("step_var")
problem_names.append("step_vararray")

problem_names = ['convex_var']

for problem_name in problem_names:
    p = __import__(problem_name)

    model = p.define_model(**kwds)
    inst = model.create()

    results = opt.solve(inst,tee=True)

    inst.load(results)

    res = dict()
    for block in inst.block_data_objects(active=True):
        for variable in itervalues(block.component_map(Var, active=True)):
            for var in itervalues(variable):
                name = var.name
                if (name[:2] == 'Fx') or (name[:1] == 'x'):
                    res[name] = value(var)
    print(res)

    #with open(problem_name+'_baseline_results.yml','w') as f:
    #    yaml.dump(res,f)
