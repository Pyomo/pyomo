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
from pyomo.dae import *
from disease_DAE import model

instance = model.create_instance('disease.dat')

discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(instance, nfe=520, ncp=3)


def _S_bar(model):
    return model.S_bar == sum(model.S[i] for i in model.TIME if i != 0) / (
        len(model.TIME) - 1
    )


instance.con_S_bar = Constraint(rule=_S_bar)

solver = SolverFactory('ipopt')
results = solver.solve(instance, tee=True)
