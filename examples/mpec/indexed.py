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

from pyomo.environ import *
from pyomo.mpec import *


n = 5

model = ConcreteModel()

model.x = Var(RangeSet(1, n))

model.f = Objective(expr=sum(i * (model.x[i] - 1) ** 2 for i in range(1, n + 1)))


def compl_(model, i):
    return complements(model.x[i] >= 0, model.x[i + 1] >= 0)


model.compl = Complementarity(RangeSet(1, n - 1), rule=compl_)
