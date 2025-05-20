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
#
# Test behavior of concrete classes.
#

import json
import os
from os.path import abspath, dirname, join

currdir = dirname(abspath(__file__)) + os.sep

import pyomo.common.unittest as unittest

from pyomo.opt import check_available_solvers
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory

solvers = check_available_solvers('glpk')


@unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
class Test(unittest.TestCase):
    def test_blending(self):
        """The blending example from the PuLP documentation"""
        model = ConcreteModel()

        model.x1 = Var(bounds=(0, None), doc="ChickenPercent")
        model.x2 = Var(bounds=(0, None), doc="BeefPercent")

        model.obj = Objective(
            expr=0.013 * model.x1 + 0.008 * model.x2,
            doc="Total Cost of Ingredients per can",
        )

        model.c0 = Constraint(expr=model.x1 + model.x2 == 100.0, doc="Percentage Sum")
        model.c1 = Constraint(
            expr=0.100 * model.x1 + 0.200 * model.x2 >= 8.0, doc="Protein Requirement"
        )
        model.c2 = Constraint(
            expr=0.080 * model.x1 + 0.100 * model.x2 >= 6.0, doc="Fat Requirement"
        )
        model.c3 = Constraint(
            expr=0.001 * model.x1 + 0.005 * model.x2 <= 2.0, doc="Fiber Requirement"
        )
        model.c4 = Constraint(
            expr=0.002 * model.x1 + 0.005 * model.x2 <= 0.4, doc="Salt Requirement"
        )
        opt = SolverFactory('glpk')
        results = opt.solve(model)
        model.solutions.store_to(results)
        results.write(filename=join(currdir, "blend.out"), format='json')
        with (
            open(join(currdir, "blend.out"), 'r') as out,
            open(join(currdir, "blend.txt"), 'r') as txt,
        ):
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-2, allow_second_superset=True
            )


if __name__ == "__main__":
    unittest.main()
