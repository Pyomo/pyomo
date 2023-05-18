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

from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.common.unittest as unittest
from pyomo.contrib.doe import (
    MeasurementVariables,
    DesignVariables,
    ScenarioGenerator,
    DesignOfExperiments,
    VariablesWithIndices,
)
from pyomo.contrib.doe.examples.reactor_kinetics import create_model, disc_for_measure


class TestMeasurementError(unittest.TestCase):
    def test(self):
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        variable_name = "C"
        indices = {0: ['CA', 'CB', 'CC'], 1: t_control}
        # measurement object
        measurements = MeasurementVariables()
        # if time index is not in indices, an value error is thrown.
        with self.assertRaises(ValueError):
            measurements.add_variables(
                variable_name, indices=indices, time_index_position=2
            )


class TestDesignError(unittest.TestCase):
    def test(self):
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        # design object
        exp_design = DesignVariables()

        # add T as design variable
        var_T = 'T'
        indices_T = {0: t_control}
        exp1_T = [470, 300, 300, 300, 300, 300, 300, 300, 300]

        upper_bound = [
            700,
            700,
            700,
            700,
            700,
            700,
            700,
            700,
            700,
            800,
        ]  # wrong upper bound since it has more elements than the length of variable names
        lower_bound = [300, 300, 300, 300, 300, 300, 300, 300, 300]

        with self.assertRaises(ValueError):
            exp_design.add_variables(
                var_T,
                indices=indices_T,
                time_index_position=0,
                values=exp1_T,
                lower_bounds=lower_bound,
                upper_bounds=upper_bound,
            )


@unittest.skipIf(not numpy_available, "Numpy is not available")
class TestPriorFIMError(unittest.TestCase):
    def test(self):
        # Control time set [h]
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
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

        parameter_dict = {"A1": 1, "A2": 1, "E1": 1}

        # empty prior
        prior_right = [[0] * 3 for i in range(3)]
        prior_pass = [[0] * 5 for i in range(10)]

        # check if the error can be thrown when given a wrong shape of FIM prior
        with self.assertRaises(ValueError):
            doe_object = DesignOfExperiments(
                parameter_dict,
                exp_design,
                measurements,
                create_model,
                prior_FIM=prior_pass,
                discretize_model=disc_for_measure,
            )


class TestMeasurement(unittest.TestCase):
    """Test the MeasurementVariables class, specify, add_element, update_variance, check_subset functions."""

    def test_setup(self):
        ### add_element function

        # control time for C [h]
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        # control time for T [h]
        t_control2 = [0.2, 0.4, 0.6, 0.8]

        # measurement object
        measurements = MeasurementVariables()

        # add variable C
        variable_name = "C"
        indices = {0: ['CA', 'CB', 'CC'], 1: t_control}
        measurements.add_variables(
            variable_name, indices=indices, time_index_position=1
        )

        # add variable T
        variable_name2 = "T"
        indices2 = {0: [1, 3, 5], 1: t_control2}
        measurements.add_variables(
            variable_name2, indices=indices2, time_index_position=1, variance=10
        )

        # check variable names
        self.assertEqual(measurements.variable_names[0], 'C[CA,0]')
        self.assertEqual(measurements.variable_names[1], 'C[CA,0.125]')
        self.assertEqual(measurements.variable_names[-1], 'T[5,0.8]')
        self.assertEqual(measurements.variable_names[-2], 'T[5,0.6]')
        self.assertEqual(measurements.variance['T[5,0.4]'], 10)
        self.assertEqual(measurements.variance['T[5,0.6]'], 10)
        self.assertEqual(measurements.variance['T[5,0.4]'], 10)
        self.assertEqual(measurements.variance['T[5,0.6]'], 10)

        ### specify function
        var_names = [
            'C[CA,0]',
            'C[CA,0.125]',
            'C[CA,0.875]',
            'C[CA,1]',
            'C[CB,0]',
            'C[CB,0.125]',
            'C[CB,0.25]',
            'C[CB,0.375]',
            'C[CC,0]',
            'C[CC,0.125]',
            'C[CC,0.25]',
            'C[CC,0.375]',
        ]

        measurements2 = MeasurementVariables()
        measurements2.set_variable_name_list(var_names)

        self.assertEqual(measurements2.variable_names[1], 'C[CA,0.125]')
        self.assertEqual(measurements2.variable_names[-1], 'C[CC,0.375]')

        ### check_subset function
        self.assertTrue(measurements.check_subset(measurements2))


class TestDesignVariable(unittest.TestCase):
    """Test the DesignVariable class, specify, add_element, add_bounds, update_values."""

    def test_setup(self):
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]

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

        self.assertEqual(
            exp_design.variable_names,
            [
                'CA0[0]',
                'T[0]',
                'T[0.125]',
                'T[0.25]',
                'T[0.375]',
                'T[0.5]',
                'T[0.625]',
                'T[0.75]',
                'T[0.875]',
                'T[1]',
            ],
        )
        self.assertEqual(exp_design.variable_names_value['CA0[0]'], 5)
        self.assertEqual(exp_design.variable_names_value['T[0]'], 470)
        self.assertEqual(exp_design.upper_bounds['CA0[0]'], 5)
        self.assertEqual(exp_design.upper_bounds['T[0]'], 700)
        self.assertEqual(exp_design.lower_bounds['CA0[0]'], 1)
        self.assertEqual(exp_design.lower_bounds['T[0]'], 300)

        design_names = exp_design.variable_names
        exp1 = [4, 600, 300, 300, 300, 300, 300, 300, 300, 300]
        exp1_design_dict = dict(zip(design_names, exp1))
        exp_design.update_values(exp1_design_dict)
        self.assertEqual(exp_design.variable_names_value['CA0[0]'], 4)
        self.assertEqual(exp_design.variable_names_value['T[0]'], 600)


class TestParameter(unittest.TestCase):
    """Test the ScenarioGenerator class, generate_scenario function."""

    def test_setup(self):
        # set up parameter class
        param_dict = {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}

        scenario_gene = ScenarioGenerator(param_dict, formula="central", step=0.1)
        parameter_set = scenario_gene.ScenarioData

        self.assertAlmostEqual(parameter_set.eps_abs['A1'], 16.9582, places=1)
        self.assertAlmostEqual(parameter_set.eps_abs['E1'], 1.5554, places=1)
        self.assertEqual(parameter_set.scena_num['A2'], [2, 3])
        self.assertEqual(parameter_set.scena_num['E1'], [4, 5])
        self.assertAlmostEqual(parameter_set.scenario[0]['A1'], 93.2699, places=1)
        self.assertAlmostEqual(parameter_set.scenario[2]['A2'], 408.8895, places=1)
        self.assertAlmostEqual(parameter_set.scenario[-1]['E2'], 13.54, places=1)
        self.assertAlmostEqual(parameter_set.scenario[-2]['E2'], 16.55, places=1)


class TestVariablesWithIndices(unittest.TestCase):
    """Test the DesignVariable class, specify, add_element, add_bounds, update_values."""

    def test_setup(self):
        special = VariablesWithIndices()
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        ### add_element function
        # add CAO as design variable
        var_C = 'CA0'
        indices_C = {0: [0]}
        exp1_C = [5]
        special.add_variables(
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

        special.add_variables(
            var_T,
            indices=indices_T,
            time_index_position=0,
            values=exp1_T,
            lower_bounds=300,
            upper_bounds=700,
        )

        self.assertEqual(
            special.variable_names,
            [
                'CA0[0]',
                'T[0]',
                'T[0.125]',
                'T[0.25]',
                'T[0.375]',
                'T[0.5]',
                'T[0.625]',
                'T[0.75]',
                'T[0.875]',
                'T[1]',
            ],
        )
        self.assertEqual(special.variable_names_value['CA0[0]'], 5)
        self.assertEqual(special.variable_names_value['T[0]'], 470)
        self.assertEqual(special.upper_bounds['CA0[0]'], 5)
        self.assertEqual(special.upper_bounds['T[0]'], 700)
        self.assertEqual(special.lower_bounds['CA0[0]'], 1)
        self.assertEqual(special.lower_bounds['T[0]'], 300)


if __name__ == '__main__':
    unittest.main()
