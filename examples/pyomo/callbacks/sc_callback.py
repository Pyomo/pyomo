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


from pyomo.core import *
from sc import *


@pyomo_callback('solve-callback')
def solve_callback(solver, model):
    print("CB-Solve")


@pyomo_callback('cut-callback')
def cut_callback(solver, model):
    print("CB-Cut")


@pyomo_callback('node-callback')
def node_callback(solver, model):
    print("CB-Node")
