#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#
#  Pyomo.DoE was produced under the Department of Energy Carbon Capture Simulation
#  Initiative (CCSI), and is copyright (c) 2022 by the software owners:
#  TRIAD National Security, LLC., Lawrence Livermore National Security, LLC.,
#  Lawrence Berkeley National Laboratory, Pacific Northwest National Laboratory,
#  Battelle Memorial Institute, University of Notre Dame,
#  The University of Pittsburgh, The University of Texas at Austin,
#  University of Toledo, West Virginia University, et al. All rights reserved.
#
#  NOTICE. This Software was developed under funding from the
#  U.S. Department of Energy and the U.S. Government consequently retains
#  certain rights. As such, the U.S. Government has been granted for itself
#  and others acting on its behalf a paid-up, nonexclusive, irrevocable,
#  worldwide license in the Software to reproduce, distribute copies to the
#  public, prepare derivative works, and perform publicly and display
#  publicly, and to permit other to do so.
#  ___________________________________________________________________________


# import libraries
from pyomo.common.dependencies import numpy as np, numpy_available, pandas_available
import pyomo.common.unittest as unittest
from pyomo.contrib.doe import DesignOfExperiments, MeasurementVariables, DesignVariables
from pyomo.environ import value, ConcreteModel
from pyomo.contrib.doe.examples.reactor_kinetics import create_model, disc_for_measure
from pyomo.opt import SolverFactory

ipopt_available = SolverFactory('ipopt').available()


class Test_example_options(unittest.TestCase):
    """Test the three options in the kinetics example."""

    def test_setUP(self):
        # parmest option
        mod = create_model(model_option="parmest")

        # global and block option
        mod = ConcreteModel()
        create_model(mod, model_option="stage1")
        create_model(mod, model_option="stage2")
        # both options need a given model, or raise errors
        with self.assertRaises(ValueError):
            create_model(model_option="stage1")

        with self.assertRaises(ValueError):
            create_model(model_option="stage2")

        with self.assertRaises(ValueError):
            create_model(model_option="NotDefine")


class Test_doe_object(unittest.TestCase):
    """Test the kinetics example with both the sequential_finite mode and the direct_kaug mode"""

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    @unittest.skipIf(not pandas_available, "Pandas is not available")
    def test_setUP(self):
        ### Define inputs
        # Control time set [h]
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        # Define parameter nominal value
        parameter_dict = {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}

        # measurement object
        variable_name = "C"
        indices = {0: ['CA', 'CB', 'CC'], 1: t_control}

        measurements = MeasurementVariables()
        measurements.add_variables(
            variable_name, indices=indices, time_index_position=1
        )

        # design object
        exp_design = DesignVariables()

        # add CAO as design variable
        var_C = 'CA0'
        indices_C = {0: [0]}
        exp1_C = [5]
        exp_design.add_variables(
            var_C,
            indices=indices_C,
            time_index_position=0,
            values=exp1_C,
            lower_bounds=1,
            upper_bounds=5,
        )

        # add T as design variable
        var_T = 'T'
        indices_T = {0: t_control}
        exp1_T = [470, 300, 300, 300, 300, 300, 300, 300, 300]

        exp_design.add_variables(
            var_T,
            indices=indices_T,
            time_index_position=0,
            values=exp1_T,
            lower_bounds=300,
            upper_bounds=700,
        )

        ### Test sequential_finite mode
        sensi_opt = "sequential_finite"

        design_names = exp_design.variable_names
        exp1 = [5, 570, 300, 300, 300, 300, 300, 300, 300, 300]
        exp1_design_dict = dict(zip(design_names, exp1))

        exp_design.update_values(exp1_design_dict)

        doe_object = DesignOfExperiments(
            parameter_dict,
            exp_design,
            measurements,
            create_model,
            discretize_model=disc_for_measure,
        )

        result = doe_object.compute_FIM(
            mode=sensi_opt, scale_nominal_param_value=True, formula="central"
        )

        result.result_analysis()

        self.assertAlmostEqual(np.log10(result.trace), 2.7885, places=2)
        self.assertAlmostEqual(np.log10(result.det), 2.8218, places=2)
        self.assertAlmostEqual(np.log10(result.min_eig), -1.0123, places=2)

        ### check subset feature
        sub_name = "C"
        sub_indices = {0: ["CB", "CC"], 1: [0.125, 0.25, 0.5, 0.75, 0.875]}

        measure_subset = MeasurementVariables()
        measure_subset.add_variables(
            sub_name, indices=sub_indices, time_index_position=1
        )
        sub_result = result.subset(measure_subset)
        sub_result.result_analysis()

        self.assertAlmostEqual(np.log10(sub_result.trace), 2.5535, places=2)
        self.assertAlmostEqual(np.log10(sub_result.det), 1.3464, places=2)
        self.assertAlmostEqual(np.log10(sub_result.min_eig), -1.5386, places=2)

        ### Test direct_kaug mode
        sensi_opt = "direct_kaug"
        # Define a new experiment

        exp1 = [5, 570, 400, 300, 300, 300, 300, 300, 300, 300]
        exp1_design_dict = dict(zip(design_names, exp1))
        exp_design.update_values(exp1_design_dict)

        doe_object = DesignOfExperiments(
            parameter_dict,
            exp_design,
            measurements,
            create_model,
            discretize_model=disc_for_measure,
        )

        result = doe_object.compute_FIM(
            mode=sensi_opt, scale_nominal_param_value=True, formula="central"
        )

        result.result_analysis()

        self.assertAlmostEqual(np.log10(result.trace), 2.7211, places=2)
        self.assertAlmostEqual(np.log10(result.det), 2.0845, places=2)
        self.assertAlmostEqual(np.log10(result.min_eig), -1.3510, places=2)

        ### Test stochastic_program mode

        exp1 = [5, 570, 300, 300, 300, 300, 300, 300, 300, 300]
        exp1_design_dict = dict(zip(design_names, exp1))
        exp_design.update_values(exp1_design_dict)

        # add a prior information (scaled FIM with T=500 and T=300 experiments)
        prior = np.asarray(
            [
                [28.67892806, 5.41249739, -81.73674601, -24.02377324],
                [5.41249739, 26.40935036, -12.41816477, -139.23992532],
                [-81.73674601, -12.41816477, 240.46276004, 58.76422806],
                [-24.02377324, -139.23992532, 58.76422806, 767.25584508],
            ]
        )

        doe_object2 = DesignOfExperiments(
            parameter_dict,
            exp_design,
            measurements,
            create_model,
            prior_FIM=prior,
            discretize_model=disc_for_measure,
        )

        square_result, optimize_result = doe_object2.stochastic_program(
            if_optimize=True,
            if_Cholesky=True,
            scale_nominal_param_value=True,
            objective_option="det",
            L_initial=np.linalg.cholesky(prior),
        )

        self.assertAlmostEqual(value(optimize_result.model.CA0[0]), 5.0, places=2)
        self.assertAlmostEqual(value(optimize_result.model.T[0.5]), 300, places=2)


if __name__ == '__main__':
    unittest.main()
