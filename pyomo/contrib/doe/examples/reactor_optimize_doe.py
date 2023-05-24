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

from pyomo.common.dependencies import numpy as np
from pyomo.contrib.doe.examples.reactor_kinetics import create_model, disc_for_measure
from pyomo.contrib.doe import DesignOfExperiments, MeasurementVariables, DesignVariables


def main():
    ### Define inputs
    # Control time set [h]
    t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
    # Define parameter nominal value
    parameter_dict = {"A1": 85, "A2": 372, "E1": 8, "E2": 15}

    # measurement object
    measurements = MeasurementVariables()
    measurements.add_variables(
        "C",  # name of measurement
        indices={0: ["CA", "CB", "CC"], 1: t_control},  # indices of measurement
        time_index_position=1,
    )  # position of time index

    # design object
    exp_design = DesignVariables()

    # add CAO as design variable
    exp_design.add_variables(
        "CA0",  # name of design variable
        indices={0: [0]},  # indices of design variable
        time_index_position=0,  # position of time index
        values=[5],  # nominal value of design variable
        lower_bounds=1,  # lower bound of design variable
        upper_bounds=5,  # upper bound of design variable
    )

    # add T as design variable
    exp_design.add_variables(
        "T",  # name of design variable
        indices={0: t_control},  # indices of design variable
        time_index_position=0,  # position of time index
        values=[
            470,
            300,
            300,
            300,
            300,
            300,
            300,
            300,
            300,
        ],  # nominal value of design variable
        lower_bounds=300,  # lower bound of design variable
        upper_bounds=700,  # upper bound of design variable
    )

    design_names = exp_design.variable_names
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
        parameter_dict,  # dictionary of parameters
        exp_design,  # design variables
        measurements,  # measurement variables
        create_model,  # function to create model
        prior_FIM=prior,  # prior information
        discretize_model=disc_for_measure,  # function to discretize model
    )

    square_result, optimize_result = doe_object2.stochastic_program(
        if_optimize=True,  # if optimize
        if_Cholesky=True,  # if use Cholesky decomposition
        scale_nominal_param_value=True,  # if scale nominal parameter value
        objective_option="det",  # objective option
        L_initial=np.linalg.cholesky(prior),  # initial Cholesky decomposition
    )

    square_result, optimize_result = doe_object2.stochastic_program(
        if_optimize=True,  # if optimize
        if_Cholesky=True,  # if use Cholesky decomposition
        scale_nominal_param_value=True,  # if scale nominal parameter value
        objective_option="trace",  # objective option
        L_initial=np.linalg.cholesky(prior),  # initial Cholesky decomposition
    )


if __name__ == "__main__":
    main()
