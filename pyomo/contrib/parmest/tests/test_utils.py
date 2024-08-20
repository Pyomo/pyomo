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

from pyomo.common.dependencies import pandas as pd, pandas_available

from pyomo.core.base.var import IndexedVar
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestUtils(unittest.TestCase):

    def test_convert_param_to_var(self):
        # TODO: Check that this works for different structured models (indexed, blocks, etc)

        # test params
        #############

        from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
            ReactorDesignExperiment,
        )

        data = pd.DataFrame(
            data=[
                [1.05, 10000, 3458.4, 1060.8, 1683.9, 1898.5],
                [1.10, 10000, 3535.1, 1064.8, 1613.3, 1893.4],
                [1.15, 10000, 3609.1, 1067.8, 1547.5, 1887.8],
            ],
            columns=["sv", "caf", "ca", "cb", "cc", "cd"],
        )

        # make model
        exp = ReactorDesignExperiment(data, 0)
        instance = exp.get_labeled_model()

        param_CUIDs = list(instance.unknown_parameters.values())
        m_vars = parmest.utils.convert_params_to_vars(
            instance, param_CUIDs, fix_vars=True
        )

        for v in [str(CUID) for CUID in param_CUIDs]:
            self.assertTrue(hasattr(m_vars, v))
            c = m_vars.find_component(v)
            self.assertIsInstance(c, pyo.Var)
            self.assertTrue(c.fixed)
            c_old = instance.find_component(v)
            self.assertEqual(pyo.value(c), pyo.value(c_old))
            self.assertTrue(c in m_vars.unknown_parameters)

        # test indexed params
        #####################

        from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
            RooneyBieglerExperiment,
        )

        self.data = pd.DataFrame(
            data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
            columns=["hour", "y"],
        )

        def rooney_biegler_indexed_params(data):
            model = pyo.ConcreteModel()

            model.param_names = pyo.Set(initialize=["asymptote", "rate_constant"])
            model.theta = pyo.Param(
                model.param_names,
                initialize={"asymptote": 15, "rate_constant": 0.5},
                mutable=True,
            )

            model.hour = pyo.Param(within=pyo.PositiveReals, mutable=True)
            model.y = pyo.Param(within=pyo.PositiveReals, mutable=True)

            def response_rule(m, h):
                expr = m.theta["asymptote"] * (
                    1 - pyo.exp(-m.theta["rate_constant"] * h)
                )
                return expr

            model.response_function = pyo.Expression(data.hour, rule=response_rule)

            return model

        class RooneyBieglerExperimentIndexedParams(RooneyBieglerExperiment):

            def create_model(self):
                data_df = self.data.to_frame().transpose()
                self.model = rooney_biegler_indexed_params(data_df)

            def label_model(self):

                m = self.model

                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update(
                    [(m.hour, self.data["hour"]), (m.y, self.data["y"])]
                )

                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update((k, pyo.ComponentUID(k)) for k in [m.theta])

        exp = RooneyBieglerExperimentIndexedParams(self.data.loc[0, :])
        instance = exp.get_labeled_model()

        param_CUIDs = list(instance.unknown_parameters.values())
        m_vars = parmest.utils.convert_params_to_vars(
            instance, param_CUIDs, fix_vars=True
        )

        for v in [str(CUID) for CUID in param_CUIDs]:
            self.assertTrue(hasattr(m_vars, v))
            c = m_vars.find_component(v)
            self.assertIsInstance(c, IndexedVar)
            for _, iv in c.items():
                self.assertTrue(iv.fixed)
                iv_old = instance.find_component(iv)
                self.assertEqual(pyo.value(iv), pyo.value(iv_old))
            self.assertTrue(c in m_vars.unknown_parameters)

        # test hierarchical model
        #########################

        m = pyo.ConcreteModel()
        m.p1 = pyo.Param(initialize=1, mutable=True)
        m.b = pyo.Block()
        m.b.p2 = pyo.Param(initialize=2, mutable=True)

        param_CUIDs = [pyo.ComponentUID(m.p1), pyo.ComponentUID(m.b.p2)]
        m_vars = parmest.utils.convert_params_to_vars(m, param_CUIDs)

        for v in [str(CUID) for CUID in param_CUIDs]:
            c = m_vars.find_component(v)
            self.assertIsInstance(c, pyo.Var)
            c_old = m.find_component(v)
            self.assertEqual(pyo.value(c), pyo.value(c_old))

if __name__ == "__main__":
    unittest.main()
