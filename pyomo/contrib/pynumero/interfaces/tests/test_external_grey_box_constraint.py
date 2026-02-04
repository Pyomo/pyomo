#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.examples.external_grey_box.react_example.reactor_model_residuals import (
    ReactorModelWithHessian,
)

from pyomo.contrib.pynumero.interfaces.external_grey_box_constraint import ExternalGreyBoxConstraint


def build_model():
    # in this simple example, we will use an external grey box model representing
    # a steady-state reactor, and solve for the space velocity that maximizes
    # the ratio of B to the other components coming out of the reactor
    # This example illustrates the use of "equality constraints" or residuals
    # in the external grey box example as well as outputs
    m = pyo.ConcreteModel()

    # create a block to store the external reactor model
    m.reactor = ExternalGreyBoxBlock(external_model=ReactorModelWithHessian())

    m.reactor.con1 = ExternalGreyBoxConstraint(implicit_constraint_id="ca_bal")

    m.cafcon = pyo.Constraint(expr=m.reactor.inputs['caf'] == 10000)
    m.obj = pyo.Objective(expr=m.reactor.outputs['cb_ratio'], sense=pyo.maximize)

    return m


def test_build_model():
    m = build_model()
    assert False


if __name__ == "__main__":
    m = build_model()

    solver = pyo.SolverFactory("cyipopt")
    solver.solve(m, tee=True)

    m.reactor.con1.display()
    print(type(m.reactor.con1))
    print(m.reactor.con1._implicit_constraint_id)
    print(pyo.value(m.reactor.con1.lb))
    print(pyo.value(m.reactor.con1.ub))
    print(pyo.value(m.reactor.con1.body))
    print(m.reactor.con1.active)
