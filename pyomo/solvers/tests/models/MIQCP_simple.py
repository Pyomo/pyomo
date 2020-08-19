#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint, Binary, maximize
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

@register_model
class MIQCP_simple(_BaseTestModel):
    """
    A mixed-integer model with a quadratic objective and quadratic constraints
    """

    description = "MIQCP_simple"
    level = ('nightly', 'expensive')
    capabilities = set(['linear', 'integer', 'quadratic_constraint'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.x = Var(within=Binary)
        model.y = Var(within=Binary)
        model.z = Var(within=Binary)

        model.obj = Objective(expr=model.x,sense=maximize)
        model.c0 = Constraint(expr=model.x+model.y+model.z == 1)
        model.qc0 = Constraint(expr=model.x**2 + model.y**2 <= model.z**2)
        model.qc1 = Constraint(expr=model.x**2 <= model.y*model.z)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = None
        model.y.value = None
        model.z.value = None

@register_model
class MIQCP_simple_kernel(MIQCP_simple):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description

        model.x = pmo.variable(domain=Binary)
        model.y = pmo.variable(domain=Binary)
        model.z = pmo.variable(domain=Binary)

        model.obj = pmo.objective(model.x,sense=maximize)
        model.c0 = pmo.constraint(model.x+model.y+model.z == 1)
        model.qc0 = pmo.constraint(model.x**2 + model.y**2 <= model.z**2)
        model.qc1 = pmo.constraint(model.x**2 <= model.y*model.z)
