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


import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.doe import ModelOptionLib


def disc_for_measure(m, nfe=32, block=True):
    """Pyomo.DAE discretization

    Arguments
    ---------
    m: Pyomo model
    nfe: number of finite elements b
    block: if True, the input model has blocks
    """
    discretizer = pyo.TransformationFactory("dae.collocation")
    if block:
        for s in range(len(m.block)):
            discretizer.apply_to(m.block[s], nfe=nfe, ncp=3, wrt=m.block[s].t)
    else:
        discretizer.apply_to(m, nfe=nfe, ncp=3, wrt=m.t)
    return m


def create_model(
    mod=None,
    model_option="stage2",
    control_time=[0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
    control_val=None,
    t_range=[0.0, 1],
    CA_init=1,
    C_init=0.1,
):
    """
    This is an example user model provided to DoE library.
    It is a dynamic problem solved by Pyomo.DAE.

    Arguments
    ---------
    mod: Pyomo model. If None, a Pyomo concrete model is created
    model_option: choose from the 3 options in model_option
        if ModelOptionLib.parmest, create a process model.
        if ModelOptionLib.stage1, create the global model.
        if ModelOptionLib.stage2, add model variables and constraints for block.
    control_time: a list of control timepoints
    control_val: control design variable values T at corresponding timepoints
    t_range: time range, h
    CA_init: time-independent design (control) variable, an initial value for CA
    C_init: An initial value for C

    Return
    ------
    m: a Pyomo.DAE model
    """

    theta = {"A1": 84.79, "A2": 371.72, "E1": 7.78, "E2": 15.05}

    model_option = ModelOptionLib(model_option)

    if model_option == ModelOptionLib.parmest:
        mod = pyo.ConcreteModel()
        return_m = True
    elif model_option == ModelOptionLib.stage1 or model_option == ModelOptionLib.stage2:
        if not mod:
            raise ValueError(
                "If model option is stage1 or stage2, a created model needs to be provided."
            )
        return_m = False
    else:
        raise ValueError(
            "model_option needs to be defined as parmest,stage1, or stage2."
        )

    if not control_val:
        control_val = [300] * 9

    controls = {}
    for i, t in enumerate(control_time):
        controls[t] = control_val[i]

    mod.t0 = pyo.Set(initialize=[0])
    mod.t_con = pyo.Set(initialize=control_time)
    mod.CA0 = pyo.Var(
        mod.t0, initialize=CA_init, bounds=(1.0, 5.0), within=pyo.NonNegativeReals
    )  # mol/L

    # check if control_time is in time range
    assert (
        control_time[0] >= t_range[0] and control_time[-1] <= t_range[1]
    ), "control time is outside time range."

    if model_option == ModelOptionLib.stage1:
        mod.T = pyo.Var(
            mod.t_con,
            initialize=controls,
            bounds=(300, 700),
            within=pyo.NonNegativeReals,
        )
        return

    else:
        para_list = ["A1", "A2", "E1", "E2"]

        ### Add variables
        mod.CA_init = CA_init
        mod.para_list = para_list

        # timepoints
        mod.t = ContinuousSet(bounds=t_range, initialize=control_time)

        # time-dependent design variable, initialized with the first control value
        def T_initial(m, t):
            if t in m.t_con:
                return controls[t]
            else:
                # count how many control points are before the current t;
                # locate the nearest neighbouring control point before this t
                neighbour_t = max(tc for tc in control_time if tc < t)
                return controls[neighbour_t]

        mod.T = pyo.Var(
            mod.t, initialize=T_initial, bounds=(300, 700), within=pyo.NonNegativeReals
        )

        mod.R = 8.31446261815324  # J / K / mole

        # Define parameters as Param
        mod.A1 = pyo.Var(initialize=theta["A1"])
        mod.A2 = pyo.Var(initialize=theta["A2"])
        mod.E1 = pyo.Var(initialize=theta["E1"])
        mod.E2 = pyo.Var(initialize=theta["E2"])

        # Concentration variables under perturbation
        mod.C_set = pyo.Set(initialize=["CA", "CB", "CC"])
        mod.C = pyo.Var(
            mod.C_set, mod.t, initialize=C_init, within=pyo.NonNegativeReals
        )

        # time derivative of C
        mod.dCdt = DerivativeVar(mod.C, wrt=mod.t)

        # kinetic parameters
        def kp1_init(m, t):
            return m.A1 * pyo.exp(-m.E1 * 1000 / (m.R * m.T[t]))

        def kp2_init(m, t):
            return m.A2 * pyo.exp(-m.E2 * 1000 / (m.R * m.T[t]))

        mod.kp1 = pyo.Var(mod.t, initialize=kp1_init)
        mod.kp2 = pyo.Var(mod.t, initialize=kp2_init)

        def T_control(m, t):
            """
            T at interval timepoint equal to the T of the control time point at the beginning of this interval
            Count how many control points are before the current t;
            locate the nearest neighbouring control point before this t
            """
            if t in m.t_con:
                return pyo.Constraint.Skip
            else:
                neighbour_t = max(tc for tc in control_time if tc < t)
                return m.T[t] == m.T[neighbour_t]

        def cal_kp1(m, t):
            """
            Create the perturbation parameter sets
            m: model
            t: time
            """
            # LHS: 1/h
            # RHS: 1/h*(kJ/mol *1000J/kJ / (J/mol/K) / K)
            return m.kp1[t] == m.A1 * pyo.exp(-m.E1 * 1000 / (m.R * m.T[t]))

        def cal_kp2(m, t):
            """
            Create the perturbation parameter sets
            m: model
            t: time
            """
            # LHS: 1/h
            # RHS: 1/h*(kJ/mol *1000J/kJ / (J/mol/K) / K)
            return m.kp2[t] == m.A2 * pyo.exp(-m.E2 * 1000 / (m.R * m.T[t]))

        def dCdt_control(m, y, t):
            """
            Calculate CA in Jacobian matrix analytically
            y: CA, CB, CC
            t: timepoints
            """
            if y == "CA":
                return m.dCdt[y, t] == -m.kp1[t] * m.C["CA", t]
            elif y == "CB":
                return m.dCdt[y, t] == m.kp1[t] * m.C["CA", t] - m.kp2[t] * m.C["CB", t]
            elif y == "CC":
                return pyo.Constraint.Skip

        def alge(m, t):
            """
            The algebraic equation for mole balance
            z: m.pert
            t: time
            """
            return m.C["CA", t] + m.C["CB", t] + m.C["CC", t] == m.CA0[0]

        # Control time
        mod.T_rule = pyo.Constraint(mod.t, rule=T_control)

        # calculating C, Jacobian, FIM
        mod.k1_pert_rule = pyo.Constraint(mod.t, rule=cal_kp1)
        mod.k2_pert_rule = pyo.Constraint(mod.t, rule=cal_kp2)
        mod.dCdt_rule = pyo.Constraint(mod.C_set, mod.t, rule=dCdt_control)

        mod.alge_rule = pyo.Constraint(mod.t, rule=alge)

        # B.C.
        mod.C["CB", 0.0].fix(0.0)
        mod.C["CC", 0.0].fix(0.0)

        if return_m:
            return mod
