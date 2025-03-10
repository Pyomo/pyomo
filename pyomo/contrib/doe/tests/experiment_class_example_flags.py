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
# === Required imports ===
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator

from pyomo.contrib.parmest.experiment import Experiment

import itertools

# ========================


def expand_model_components(m, base_components, index_sets):
    """
    Takes model components and index sets and returns the
    model component labels.

    Arguments
    ---------
    m: Pyomo model
    base_components: list of variables from model 'm'
    index_sets: list, same length as base_components, where each
                element is a list of index sets, or None
    """
    for val, indexes in itertools.zip_longest(base_components, index_sets):
        # If the variable has no index,
        # add just the model component
        if not val.is_indexed():
            yield val
        # If the component is indexed but no
        # index supplied, add all indices
        elif indexes is None:
            yield from val.values()
        else:
            for j in itertools.product(*indexes):
                yield val[j]


class BadExperiment(object):
    def __init__(self):
        self.model = None


class ReactorExperiment(Experiment):
    def __init__(self, data, nfe, ncp):
        self.data = data
        self.nfe = nfe
        self.ncp = ncp
        self.model = None

    def get_labeled_model(self, flag=0):
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_experiment(flag=flag)
        return self.model

    def create_model(self):
        """
        This is an example user model provided to DoE library.
        It is a dynamic problem solved by Pyomo.DAE.

        Return
        ------
        m: a Pyomo.DAE model
        """

        m = self.model = pyo.ConcreteModel()

        # Model parameters
        m.R = pyo.Param(mutable=False, initialize=8.314)

        # Define model variables
        ########################
        # time
        m.t = ContinuousSet(bounds=[0, 1])

        # Concentrations
        m.CA = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.CB = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.CC = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Temperature
        m.T = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Arrhenius rate law equations
        m.A1 = pyo.Var(within=pyo.NonNegativeReals)
        m.E1 = pyo.Var(within=pyo.NonNegativeReals)
        m.A2 = pyo.Var(within=pyo.NonNegativeReals)
        m.E2 = pyo.Var(within=pyo.NonNegativeReals)

        # Differential variables (Conc.)
        m.dCAdt = DerivativeVar(m.CA, wrt=m.t)
        m.dCBdt = DerivativeVar(m.CB, wrt=m.t)

        ########################
        # End variable def.

        # Equation def'n
        ########################

        # Expression for rate constants
        @m.Expression(m.t)
        def k1(m, t):
            return m.A1 * pyo.exp(-m.E1 * 1000 / (m.R * m.T[t]))

        @m.Expression(m.t)
        def k2(m, t):
            return m.A2 * pyo.exp(-m.E2 * 1000 / (m.R * m.T[t]))

        # Concentration odes
        @m.Constraint(m.t)
        def CA_rxn_ode(m, t):
            return m.dCAdt[t] == -m.k1[t] * m.CA[t]

        @m.Constraint(m.t)
        def CB_rxn_ode(m, t):
            return m.dCBdt[t] == m.k1[t] * m.CA[t] - m.k2[t] * m.CB[t]

        # algebraic balance for concentration of C
        # Valid because the reaction system (A --> B --> C) is equimolar
        @m.Constraint(m.t)
        def CC_balance(m, t):
            return m.CA[0] == m.CA[t] + m.CB[t] + m.CC[t]

        ########################
        # End equation def'n

    def finalize_model(self):
        """
        Example finalize model function. There are two main tasks
        here:
            1. Extracting useful information for the model to align
               with the experiment. (Here: CA0, t_final, t_control)
            2. Discretizing the model subject to this information.

        Arguments
        ---------
        m: Pyomo model
        data: object containing vital experimental information
        nfe: number of finite elements
        ncp: number of collocation points for the finite elements
        """
        m = self.model

        # Unpacking data before simulation
        control_points = self.data["control_points"]

        m.CA[0].value = self.data["CA0"]
        m.CB[0].fix(self.data["CB0"])
        m.t.update(self.data["t_range"])
        m.t.update(control_points)
        m.A1.fix(self.data["A1"])
        m.A2.fix(self.data["A2"])
        m.E1.fix(self.data["E1"])
        m.E2.fix(self.data["E2"])

        m.CA[0].setlb(self.data["CA_bounds"][0])
        m.CA[0].setub(self.data["CA_bounds"][1])

        m.t_control = control_points

        # Discretizing the model
        discr = pyo.TransformationFactory("dae.collocation")
        discr.apply_to(m, nfe=self.nfe, ncp=self.ncp, wrt=m.t)

        # Initializing Temperature in the model
        cv = None
        for t in m.t:
            if t in control_points:
                cv = control_points[t]
            m.T[t].setlb(self.data["T_bounds"][0])
            m.T[t].setub(self.data["T_bounds"][1])
            m.T[t] = cv

        @m.Constraint(m.t - control_points)
        def T_control(m, t):
            """
            Piecewise constant Temperature between control points
            """
            neighbour_t = max(tc for tc in control_points if tc < t)
            return m.T[t] == m.T[neighbour_t]

        # sim.initialize_model()

    def label_experiment_impl(self, index_sets_meas, flag=0):
        """
        Example for annotating (labeling) the model with a
        full experiment.

        Arguments
        ---------

        """
        m = self.model
        base_comp_meas = [m.CA, m.CB, m.CC]

        if flag != 1:
            # Grab measurement labels
            m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_outputs.update(
                (k, None)
                for k in expand_model_components(m, base_comp_meas, index_sets_meas)
            )

        if flag != 2:
            # Adding no error for measurements currently
            m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            if flag == 5:
                m.measurement_error.update((m.CA[0], 1e-2) for k in range(1))
            else:
                m.measurement_error.update(
                    (k, 1e-2)
                    for k in expand_model_components(m, base_comp_meas, index_sets_meas)
                )

        if flag != 3:
            # Grab design variables
            base_comp_des = [m.CA, m.T]
            index_sets_des = [[[m.t.first()]], [m.t_control]]
            m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_inputs.update(
                (k, None)
                for k in expand_model_components(m, base_comp_des, index_sets_des)
            )

        if flag != 4:
            m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.unknown_parameters.update(
                (k, pyo.value(k)) for k in [m.A1, m.A2, m.E1, m.E2]
            )


class FullReactorExperiment(ReactorExperiment):
    def label_experiment(self, flag=0):
        m = self.model
        return self.label_experiment_impl(
            [[m.t_control], [m.t_control], [m.t_control]], flag=flag
        )


class FullReactorExperimentBad(ReactorExperiment):
    def label_experiment(self, flag=0):
        m = self.model

        self.label_experiment_impl(
            [[m.t_control], [m.t_control], [m.t_control]], flag=flag
        )

        m.bad_con_1 = pyo.Constraint(expr=m.CA[0] >= 1.0)
        m.bad_con_2 = pyo.Constraint(expr=m.CA[0] <= 0.0)

        return m
