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
    parameter_dict = {"A1": 85, "A2": 370, "E1": 8, "E2": 15}

    # Define measurement object
    measurements = MeasurementVariables()
    measurements.add_variables(
        "C",  # measurement variable name
        indices={
            0: ["CA", "CB", "CC"],
            1: t_control,
        },  # 0,1 are indices of the index sets
        time_index_position=1,
    )

    # design object
    exp_design = DesignVariables()

    # add CAO as design variable
    exp_design.add_variables(
        "CA0",  # design variable name
        indices={0: [0]},  # index dictionary
        time_index_position=0,  # time index position
        values=[5],  # design variable values
        lower_bounds=1,  # design variable lower bounds
        upper_bounds=5,  # design variable upper bounds
    )

    # add T as design variable
    exp_design.add_variables(
        "T",  # design variable name
        indices={0: t_control},  # index dictionary
        time_index_position=0,  # time index position
        values=[
            570,
            300,
            300,
            300,
            300,
            300,
            300,
            300,
            300,
        ],  # same length with t_control
        lower_bounds=300,  # design variable lower bounds
        upper_bounds=700,  # design variable upper bounds
    )

    ### Compute the FIM of a square model-based Design of Experiments problem
    doe_object = DesignOfExperiments(
        parameter_dict,  # parameter dictionary
        exp_design,  # DesignVariables object
        measurements,  # MeasurementVariables object
        create_model,  # create model function
        discretize_model=disc_for_measure,  # discretize model function
    )

    result = doe_object.compute_FIM(
        mode="sequential_finite",  # calculation mode
        scale_nominal_param_value=True,  # scale nominal parameter value
        formula="central",  # formula for finite difference
    )

    result.result_analysis()

    # test result
    relative_error = abs(np.log10(result.trace) - 2.78)
    assert relative_error < 0.01

    relative_error = abs(np.log10(result.det) - 2.99)
    assert relative_error < 0.01


if __name__ == "__main__":
    main()
