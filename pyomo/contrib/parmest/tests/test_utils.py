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

from pyomo.common.dependencies import (
    pandas as pd,
    pandas_available,
    numpy as np,
    numpy_available,
)

import os.path
import json

import pyomo.environ as pyo

from pyomo.common.fileutils import this_file_dir
import pyomo.common.unittest as unittest

import pyomo.contrib.parmest.parmest as parmest
from pyomo.opt import SolverFactory

from pyomo.contrib.parmest.utils.model_utils import update_model_from_suffix
from pyomo.contrib.doe.examples.reactor_example import (
    ReactorExperiment as FullReactorExperiment,
)

currdir = this_file_dir()
file_path = os.path.join(currdir, "..", "..", "doe", "examples", "result.json")

with open(file_path) as f:
    data_ex = json.load(f)
data_ex["control_points"] = {float(k): v for k, v in data_ex["control_points"].items()}

ipopt_available = pyo.SolverFactory("ipopt").available()


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestUtils(unittest.TestCase):

    def test_convert_param_to_var(self):
        # TODO: Check that this works for different structured models (indexed, blocks, etc)

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

        theta_names = ['k1', 'k2', 'k3']
        m_vars = parmest.utils.convert_params_to_vars(
            instance, theta_names, fix_vars=True
        )

        for v in theta_names:
            self.assertTrue(hasattr(m_vars, v))
            c = m_vars.find_component(v)
            self.assertIsInstance(c, pyo.Var)
            self.assertTrue(c.fixed)
            c_old = instance.find_component(v)
            self.assertEqual(pyo.value(c), pyo.value(c_old))
            self.assertTrue(c in m_vars.unknown_parameters)

    def test_update_model_from_suffix_experiment_outputs(self):
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
        experiment = ReactorDesignExperiment(data, 0)
        test_model = experiment.get_labeled_model()

        suffix_obj = test_model.experiment_outputs  # a Suffix
        var_list = list(suffix_obj.keys())  # components
        orig_var_vals = np.array([pyo.value(v) for v in var_list])
        orig_suffix_val = np.array([tag for _, tag in suffix_obj.items()])
        new_vals = orig_var_vals + 0.5
        # Update the model from the suffix
        update_model_from_suffix(suffix_obj, new_vals)
        # ── Check results ────────────────────────────────────────────────────
        new_var_vals = np.array([pyo.value(v) for v in var_list])
        new_suffix_val = np.array(list(suffix_obj.values()))
        # (1) Variables have been overwritten with `new_vals`
        self.assertTrue(np.allclose(new_var_vals, new_vals))
        # (2) Suffix tags are unchanged
        self.assertTrue(np.array_equal(new_suffix_val, orig_suffix_val))

    def test_update_model_from_suffix_measurement_error(self):
        experiment = FullReactorExperiment(data_ex, 10, 3)
        test_model = experiment.get_labeled_model()

        suffix_obj = test_model.measurement_error  # a Suffix
        var_list = list(suffix_obj.keys())  # components
        orig_var_vals = np.array([suffix_obj[v] for v in var_list])
        new_vals = orig_var_vals + 0.5
        # Update the model from the suffix
        update_model_from_suffix(suffix_obj, new_vals)
        # ── Check results ────────────────────────────────────────────────────
        new_var_vals = np.array([suffix_obj[v] for v in var_list])
        # (1) Variables have been overwritten with `new_vals`
        self.assertTrue(np.allclose(new_var_vals, new_vals))

    def test_update_model_from_suffix_length_mismatch(self):
        m = pyo.ConcreteModel()

        # Create a suffix with a Var component
        m.x = pyo.Var(initialize=0.0)
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters[m.x] = 0.0  # tag a Var
        with self.assertRaisesRegex(
            ValueError, "values length does not match suffix length"
        ):
            # Attempt to update with a list of different length
            update_model_from_suffix(m.unknown_parameters, [42, 43, 44])

    def test_update_model_from_suffix_not_numeric(self):
        m = pyo.ConcreteModel()

        # Create a suffix with a Var component
        m.x = pyo.Var(initialize=0.0)
        m.y = pyo.Var(initialize=1.0)
        bad_value = "not_a_number"
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Make multiple values
        m.unknown_parameters[m.x] = 0.0  # tag a Var
        m.unknown_parameters[m.y] = bad_value  # tag a Var with a bad value
        # Attempt to update with a list of mixed types
        # This should raise an error because this utility only allows numeric values
        # in the model to be updated.

        with self.assertRaisesRegex(
            ValueError, f"could not convert string to float: '{bad_value}'"
        ):
            # Attempt to update with a non-numeric value
            update_model_from_suffix(m.unknown_parameters, [42, bad_value])

    def test_update_model_from_suffix_wrong_component_type(self):
        m = pyo.ConcreteModel()

        # Create a suffix with a Var component
        m.x = pyo.Var(initialize=0.0)
        m.e = pyo.Expression(expr=m.x + 1)  # not Var/Param
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters[m.x] = 0.0
        m.unknown_parameters[m.e] = 1.0  # tag an Expression
        # Attempt to update with a list of wrong component type
        with self.assertRaisesRegex(
            TypeError,
            f"Unsupported component type {type(m.e)}; expected VarData or ParamData.",
        ):
            update_model_from_suffix(m.unknown_parameters, [42, 43])

    def test_update_model_from_suffix_unsupported_component(self):
        m = pyo.ConcreteModel()

        # Create a suffix with a ConstraintData component
        m.x = pyo.Var(initialize=0.0)
        m.c = pyo.Constraint(expr=m.x == 0)  # not Var/Param!

        m.bad_suffix = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.bad_suffix[m.c] = 0  # tag a Constraint

        with self.assertRaisesRegex(
            TypeError, r"Unsupported component type .*Constraint.*"
        ):
            update_model_from_suffix(m.bad_suffix, [1.0])

    def test_update_model_from_suffix_empty(self):
        m = pyo.ConcreteModel()

        # Create an empty suffix
        m.empty_suffix = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        # This should not raise an error
        update_model_from_suffix(m.empty_suffix, [])


if __name__ == "__main__":
    unittest.main()
