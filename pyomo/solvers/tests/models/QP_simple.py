#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, NonNegativeReals
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model


@register_model
class QP_simple(_BaseTestModel):
    """
    A continuous model with a quadratic objective and linear constraints
    """

    description = "QP_simple"
    level = ('nightly', 'expensive')
    capabilities = set(['linear', 'quadratic_objective'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.a = Param(initialize=1.0)
        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=NonNegativeReals)

        model.inactive_obj = Objective(expr=model.y)
        model.inactive_obj.deactivate()
        model.obj = Objective(expr=model.x**2 + 3.0*model.inactive_obj**2 + 1.0)
        model.c1 = Constraint(expr=model.a <= model.y)
        model.c2 = Constraint(expr=2.0 <= model.x/model.a - model.y <= 10)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x = 1
        model.y = 1


@register_model
class QP_simple_nosuffixes(QP_simple):

    description = "QP_simple_nosuffixes"
    test_pickling = False

    def __init__(self):
        QP_simple.__init__(self)
        self.disable_suffix_tests = True
        self.add_results("QP_simple.json")

