#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
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

# from pyomo.contrib.parmest.examples.reactor_design import reactor_design_model
# if we refactor to use the same create_model function as parmest, 
# we can just import instead of redefining the model

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.doe import ModelOptionLib, DesignOfExperiments, MeasurementVariables, DesignVariables

def create_model(
        mod=None,
        model_option="stage2"):
    
    model_option = ModelOptionLib(model_option)

    model = mod

    if model_option == ModelOptionLib.parmest:
        model = pyo.ConcreteModel()
        return_m = True
    elif model_option == ModelOptionLib.stage1 or model_option == ModelOptionLib.stage2:
        if model is None:
            raise ValueError(
                "If model option is stage1 or stage2, a created model needs to be provided."
            )
        return_m = False
    else:
        raise ValueError(
            "model_option needs to be defined as parmest, stage1, or stage2."
        )
    
    # Rate constants
    model.k1 = pyo.Var(
        initialize=5.0 / 6.0, within=pyo.PositiveReals
    )  # min^-1
    model.k2 = pyo.Var(
        initialize=5.0 / 3.0, within=pyo.PositiveReals
    )  # min^-1
    model.k3 = pyo.Var(
        initialize=1.0 / 6000.0, within=pyo.PositiveReals
    )  # m^3/(gmol min)

    # Inlet concentration of A, gmol/m^3
    model.caf = pyo.Var(initialize=10000, within=pyo.PositiveReals)

    # Space velocity (flowrate/volume)
    model.sv = pyo.Var(initialize=1.0, within=pyo.PositiveReals)

    # Outlet concentration of each component
    model.ca = pyo.Var(initialize=5000.0, within=pyo.PositiveReals)
    model.cb = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
    model.cc = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
    model.cd = pyo.Var(initialize=1000.0, within=pyo.PositiveReals)

    # Objective
    model.obj = pyo.Objective(expr=model.cb, sense=pyo.maximize)

    # Constraints
    model.ca_bal = pyo.Constraint(
        expr=(
            0
            == model.sv * model.caf
            - model.sv * model.ca
            - model.k1 * model.ca
            - 2.0 * model.k3 * model.ca**2.0
        )
    )

    model.cb_bal = pyo.Constraint(
        expr=(0 == -model.sv * model.cb + model.k1 * model.ca - model.k2 * model.cb)
    )

    model.cc_bal = pyo.Constraint(
        expr=(0 == -model.sv * model.cc + model.k2 * model.cb)
    )

    model.cd_bal = pyo.Constraint(
        expr=(0 == -model.sv * model.cd + model.k3 * model.ca**2.0)
    )

    if return_m:
        return model
    
def main():

    # measurement object
    measurements = MeasurementVariables()
    measurements.add_variables(
        "ca",
        indices=None,
        time_index_position=None
    )
    measurements.add_variables(
        "cb",
        indices=None,
        time_index_position=None
    )
    measurements.add_variables(
        "cc",
        indices=None,
        time_index_position=None
    )
    measurements.add_variables(
        "cd",
        indices=None,
        time_index_position=None
    )

    # design object
    exp_design = DesignVariables()
    exp_design.add_variables(
        "sv",
        indices=None,
        time_index_position=None,
        values=1.0,
        lower_bounds=0.1,
        upper_bounds=10.0
    )
    exp_design.add_variables(
        "caf",
        indices=None,
        time_index_position=None,
        values=10000,
        lower_bounds=5000,
        upper_bounds=15000
    )

    theta_values = {"k1": 5.0 / 6.0, "k2": 5.0 / 3.0, "k3": 1.0 / 6000.0}

    doe1 = DesignOfExperiments(
        theta_values,
        exp_design,
        measurements,
        create_model,
        prior_FIM=None
    )

    result = doe1.compute_FIM(
        mode="sequential_finite",  # calculation mode
        scale_nominal_param_value=True,  # scale nominal parameter value
        formula="central",  # formula for finite difference
    )

    result.result_analysis()

if __name__ == "__main__":
    main()

