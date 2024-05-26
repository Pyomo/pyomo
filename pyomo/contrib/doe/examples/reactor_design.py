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
from pyomo.contrib.doe import (
    ModelOptionLib,
    DesignOfExperiments,
    MeasurementVariables,
    DesignVariables,
)
from pyomo.common.dependencies import numpy as np


def create_model_legacy(mod=None, model_option=None):
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

    model = _create_model_details(model)

    if return_m:
        return model


def create_model():
    model = pyo.ConcreteModel()
    return _create_model_details(model)


def _create_model_details(model):

    # Rate constants
    model.k1 = pyo.Var(initialize=5.0 / 6.0, within=pyo.PositiveReals)  # min^-1
    model.k2 = pyo.Var(initialize=5.0 / 3.0, within=pyo.PositiveReals)  # min^-1
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

    return model


def main(legacy_create_model_interface=False):

    # measurement object
    measurements = MeasurementVariables()
    measurements.add_variables("ca", indices=None, time_index_position=None)
    measurements.add_variables("cb", indices=None, time_index_position=None)
    measurements.add_variables("cc", indices=None, time_index_position=None)
    measurements.add_variables("cd", indices=None, time_index_position=None)

    # design object
    exp_design = DesignVariables()
    exp_design.add_variables(
        "sv",
        indices=None,
        time_index_position=None,
        values=1.0,
        lower_bounds=0.1,
        upper_bounds=10.0,
    )
    exp_design.add_variables(
        "caf",
        indices=None,
        time_index_position=None,
        values=10000,
        lower_bounds=5000,
        upper_bounds=15000,
    )

    theta_values = {"k1": 5.0 / 6.0, "k2": 5.0 / 3.0, "k3": 1.0 / 6000.0}

    if legacy_create_model_interface:
        create_model_ = create_model_legacy
    else:
        create_model_ = create_model

    doe1 = DesignOfExperiments(
        theta_values, exp_design, measurements, create_model_, prior_FIM=None
    )

    result = doe1.compute_FIM(
        mode="sequential_finite",  # calculation mode
        scale_nominal_param_value=True,  # scale nominal parameter value
        formula="central",  # formula for finite difference
    )

    # doe1.model.pprint()

    result.result_analysis()

    # print("FIM =\n",result.FIM)
    # print("jac =\n",result.jaco_information)
    # print("log10 Trace of FIM: ", np.log10(result.trace))
    # print("log10 Determinant of FIM: ", np.log10(result.det))

    # test result
    expected_log10_trace = 6.815
    log10_trace = np.log10(result.trace)
    relative_error_trace = abs(log10_trace - 6.815)
    assert relative_error_trace < 0.01, (
        "log10(tr(FIM)) regression test failed, answer "
        + str(round(log10_trace, 3))
        + " does not match expected answer of "
        + str(expected_log10_trace)
    )

    expected_log10_det = 18.719
    log10_det = np.log10(result.det)
    relative_error_det = abs(log10_det - 18.719)
    assert relative_error_det < 0.01, (
        "log10(det(FIM)) regression test failed, answer "
        + str(round(log10_det, 3))
        + " does not match expected answer of "
        + str(expected_log10_det)
    )

    doe2 = DesignOfExperiments(
        theta_values, exp_design, measurements, create_model_, prior_FIM=None
    )

    square_result2, optimize_result2 = doe2.stochastic_program(
        if_optimize=True,
        if_Cholesky=True,
        scale_nominal_param_value=True,
        objective_option="det",
        jac_initial=result.jaco_information.copy(),
        step=0.1,
    )

    optimize_result2.result_analysis()
    log_det = np.log10(optimize_result2.det)
    print("log(det) = ", round(log_det, 3))
    log_det_expected = 19.266
    assert abs(log_det - log_det_expected) < 0.01, "log(det) regression test failed"

    doe3 = DesignOfExperiments(
        theta_values, exp_design, measurements, create_model_, prior_FIM=None
    )

    square_result3, optimize_result3 = doe3.stochastic_program(
        if_optimize=True,
        scale_nominal_param_value=True,
        objective_option="trace",
        jac_initial=result.jaco_information.copy(),
        step=0.1,
    )

    optimize_result3.result_analysis()
    log_trace = np.log10(optimize_result3.trace)
    log_trace_expected = 7.509
    print("log(trace) = ", round(log_trace, 3))
    assert (
        abs(log_trace - log_trace_expected) < 0.01
    ), "log(trace) regression test failed"


if __name__ == "__main__":
    main(legacy_create_model_interface=False)
