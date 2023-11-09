#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.tests import models
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import Constraint, SolverFactory, Var

from pyomo.contrib.piecewise.transform.disagreggated_logarithmic import DisaggregatedLogarithmicInnerGDPTransformation

class TestTransformPiecewiseModelToNestedInnerRepnGDP(unittest.TestCase):

    def test_solve_log_model(self):
        m = models.make_log_x_model()
        TransformationFactory(
            'contrib.piecewise.disaggregated_logarithmic'
        ).apply_to(m)
        TransformationFactory(
            'gdp.bigm'
        ).apply_to(m)
        SolverFactory('gurobi').solve(m)
        ct.check_log_x_model_soln(self, m)