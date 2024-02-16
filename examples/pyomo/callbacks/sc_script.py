#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from pyomo.opt import *
from pyomo.core import *

import sc

model = sc.pyomo_create_model()


def solve_callback(solver, model):
    print("CB-Solve")


def cut_callback(solver, model):
    print("CB-Cut")


def node_callback(solver, model):
    print("CB-Node")


instance = model.create()

opt = SolverFactory('_cplex_direct')
opt.set_callback('cut-callback', cut_callback)
opt.set_callback('node-callback', node_callback)
opt.set_callback('solve-callback', solve_callback)

results = opt.solve(instance, tee=True)
print(results)
