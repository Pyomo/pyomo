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
from pyomo.environ import SolverFactory, value, Var, Constraint, TransformationFactory
from pyomo.gdp import Disjunct
import pyomo.common.unittest as unittest
from pyomo.contrib.gdpopt.tests.four_stage_dynamic_model import build_model


class TestGDPoptLDSDA(unittest.TestCase):
    """Real unit tests for GDPopt"""

    @unittest.skipUnless(
        SolverFactory('gams').available(False)
        and SolverFactory('gams').license_is_valid(),
        "gams solver not available",
    )
    def test_solve_four_stage_dynamic_model(self):

        model = build_model(mode_transfer=True)

        # Discretize the model using dae.collocation
        discretizer = TransformationFactory('dae.collocation')
        discretizer.apply_to(model, nfe=10, ncp=3, scheme='LAGRANGE-RADAU')
        # We need to reconstruct the constraints in disjuncts after discretization.
        # This is a bug in Pyomo.dae. https://github.com/Pyomo/pyomo/issues/3101
        for disjunct in model.component_data_objects(ctype=Disjunct):
            for constraint in disjunct.component_objects(ctype=Constraint):
                constraint._constructed = False
                constraint.construct()

        for dxdt in model.component_data_objects(ctype=Var, descend_into=True):
            if 'dxdt' in dxdt.name:
                dxdt.setlb(-300)
                dxdt.setub(300)

        for direction_norm in ['L2', 'Linf']:
            result = SolverFactory('gdpopt.ldsda').solve(
                model,
                direction_norm=direction_norm,
                minlp_solver='gams',
                minlp_solver_args=dict(solver='ipopth'),
                starting_point=[1, 2],
                logical_constraint_list=[
                    model.mode_transfer_lc1,
                    model.mode_transfer_lc2,
                ],
                time_limit=100,
            )
            self.assertAlmostEqual(value(model.obj), -23.305325, places=4)


if __name__ == '__main__':
    unittest.main()
