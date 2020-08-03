#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.kernel import block, variable, objective, constraint, variable_dict, parameter, sos1
from pyomo.core import ConcreteModel, Param, Var, Objective, Constraint, NonNegativeReals, SOSConstraint, sum_product
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

@register_model
class SOS1_simple(_BaseTestModel):
    """
    A simple linear program
    """

    description = "SOS1_simple"
    level = ('nightly', 'expensive')
    capabilities = set(['linear', 'integer', 'sos1'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.a = Param(initialize=0.1)
        model.x = Var(within=NonNegativeReals)
        model.y = Var([1,2],within=NonNegativeReals)

        model.obj = Objective(expr=model.x + model.y[1]+2*model.y[2])
        model.c1 = Constraint(expr=model.a <= model.y[2])
        model.c2 = Constraint(expr=(2.0, model.x, 10.0))
        model.c3 = SOSConstraint(var=model.y, index=[1,2], sos=1)
        model.c4 = Constraint(expr=sum_product(model.y) == 1)

        # Make an empty SOSConstraint
        model.c5 = SOSConstraint(var=model.y, index=[1,2], sos=1)
        model.c5.set_items([],[])
        assert len(list(model.c5.get_items())) == 0

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = 0
        model.y[1].value = 1
        model.y[2].value = None

@register_model
class SOS1_simple_kernel(SOS1_simple):

    def _generate_model(self):
        self.model =  block()
        model = self.model
        model._name = self.description

        model.a =  parameter(value=0.1)
        model.x =  variable(domain=NonNegativeReals)
        model.y =  variable_dict()
        model.y[1] =  variable(domain=NonNegativeReals)
        model.y[2] =  variable(domain=NonNegativeReals)

        model.obj =  objective(model.x + model.y[1]+2*model.y[2])
        model.c1 =  constraint(model.a <= model.y[2])
        model.c2 =  constraint((2.0, model.x, 10.0))
        model.c3 =  sos1(model.y.values())
        model.c4 =  constraint(sum(model.y.values()) == 1)

        # Make an empty SOS constraint
        model.c5 =  sos1([])
