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

from pyomo.core.base.objective import Objective


def get_objective(block):
    obj = None
    for o in block.component_data_objects(
        Objective, descend_into=True, active=True, sort=True
    ):
        if obj is not None:
            raise ValueError('Multiple active objectives found')
        obj = o
    return obj
