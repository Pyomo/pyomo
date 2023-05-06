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


import numpy as np
import pyomo.common.unittest as unittest
from pyomo.contrib.doe.examples.reactor_kinetics import create_model, disc_for_measure
from pyomo.contrib.doe import (
    DesignOfExperiments,
    MeasurementVariables,
    DesignVariables,
    calculation_mode,
)


def main():
    ### Define inputs
    # Control time set [h]
    t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
    # Define parameter nominal value
    parameter_dict = {"A1": 85, "A2": 372, "E1": 8, "E2": 15}

    # measurement object
    measurements = MeasurementVariables()
    measurements.add_variables("C", 
                               indices= {0: ["CA", "CB", "CC"], 1: t_control}, 
                               time_index_position=1)

    # design object
    exp_design = DesignVariables()

    # add CAO as design variable
    exp_design.add_variables(
        "CA0",
        indices= {0: [0]},
        time_index_position=0,
        values=[5],
        lower_bounds=1,
        upper_bounds=5,
    )

    # add T as design variable
    exp_design.add_variables(
        "T",
        indices={0: t_control},
        time_index_position=0,
        values=[470, 300, 300, 300, 300, 300, 300, 300, 300],
        lower_bounds=300,
        upper_bounds=700,
    )

    # For each variable, we define a list of possible values that are used
    # in the sensitivity analysis

    design_ranges = {
        "CA0[0]": [1, 3, 5],
        (
            "T[0]",
            "T[0.125]",
            "T[0.25]",
            "T[0.375]",
            "T[0.5]",
            "T[0.625]",
            "T[0.75]",
            "T[0.875]",
            "T[1]",
        ): [300, 500, 700],
    }
    ## choose from "sequential_finite", "direct_kaug"
    sensi_opt = calculation_mode.direct_kaug

    doe_object = DesignOfExperiments(
        parameter_dict,
        exp_design,
        measurements,
        create_model,
        discretize_model=disc_for_measure,
    )
    # run full factorial grid search
    all_fim = doe_object.run_grid_search(design_ranges, mode=sensi_opt)

    all_fim.extract_criteria()

    ### 3 design variable example
    # Define design ranges
    design_ranges = {
        "CA0[0]": list(np.linspace(1, 5, 2)),
        "T[0]": list(np.linspace(300, 700, 2)),
        (
            "T[0.125]",
            "T[0.25]",
            "T[0.375]",
            "T[0.5]",
            "T[0.625]",
            "T[0.75]",
            "T[0.875]",
            "T[1]",
        ): [300, 500],
    }

    sensi_opt = calculation_mode.direct_kaug

    doe_object = DesignOfExperiments(
        parameter_dict,
        exp_design,
        measurements,
        create_model,
        discretize_model=disc_for_measure,
    )
    # run the grid search for 3 dimensional case
    all_fim = doe_object.run_grid_search(design_ranges, mode=sensi_opt)

    all_fim.extract_criteria()

    # see the criteria values
    all_fim.store_all_results_dataframe


if __name__ == "__main__":
    main()
