#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, NonNegativeReals, maximize, ConstraintList
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model


@register_model
class QCP_simple(_BaseTestModel):
    """
    A continuous model with a quadratic objective and quadratics constraints
    """

    description = "QCP_simple"
    level = ('nightly', 'expensive')
    capabilities = set(['linear', 'quadratic_objective', 'quadratic_constraint'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=NonNegativeReals)
        model.z = Var(within=NonNegativeReals)
        model.fixed_var = Var()
        model.fixed_var.fix(0.2)
        model.q1 = Var(bounds=(None, 0.2))
        model.q2 = Var(bounds=(-2,None))
        model.obj = Objective(expr=model.x+model.q1-model.q2,sense=maximize)
        model.c0 = Constraint(expr=model.x+model.y+model.z == 1)
        model.qc0 = Constraint(expr=model.x**2 + model.y**2 + model.fixed_var <= model.z**2)
        model.qc1 = Constraint(expr=model.x**2 <= model.y*model.z)
        model.c = ConstraintList()
        model.c.add((0, -model.q1**2 + model.fixed_var, None))
        model.c.add((None, model.q2**2 + model.fixed_var, 5))

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x = 1
        model.y = 1
        model.z = 1


@register_model
class QCP_simple_nosuffixes(QCP_simple):

    description = "QCP_simple_nosuffixes"
    test_pickling = False

    def __init__(self):
        QCP_simple.__init__(self)
        self.disable_suffix_tests = True
        self.add_results("QCP_simple.json")

