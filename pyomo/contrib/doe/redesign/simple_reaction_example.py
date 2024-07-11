# === Required imports ===
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator

import itertools
import json
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

class Experiment(object):
    def __init__(self):
        self.model = None

    def get_labeled_model(self):
        raise NotImplementedError(
            "Derived experiment class failed to implement get_labeled_model"
        )


class SimpleReactorExperiment(object):
    def __init__(self, data, nfe, ncp):
        self.data = data
        self.nfe = nfe
        self.ncp = ncp
        self.model = None

    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_experiment_impl(index_sets_meas=[[self.model.t_control], [self.model.t_control],])
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

        # Arrhenius rate law equations
        # m.A1 = pyo.Var(within=pyo.NonNegativeReals)
        m.A1 = pyo.Param(initialize=0, mutable=True)

        # Differential variables (Conc.)
        m.dCAdt = DerivativeVar(m.CA, wrt=m.t)
        # m.dCBdt = DerivativeVar(m.CB, wrt=m.t)

        ########################
        # End variable def.

        # Equation def'n
        ########################
        
        m.k1 = pyo.Var(m.t, initialize=0)

        @m.Constraint(m.t)
        def k1_con(m, t):
            return m.k1[t] == m.A1

        # @m.Expression(m.t)
        # def k1(m, t):
        #     return m.A1

        # Concentration odes
        @m.Constraint(m.t)
        def CA_rxn_ode(m, t):
            return m.dCAdt[t] == -m.k1[t] * m.CA[t]
        
        # @m.Constraint(m.t)
        # def CB_rxn_ode(m, t):
        #     return m.dCBdt[t] == m.A1 * m.CB[t]

        # algebraic balance for concentration of B
        # Valid because the reaction system (A --> B) is equimolar
        @m.Constraint(m.t)
        def CB_balance(m, t):
            return m.CA[0] == m.CA[t] + m.CB[t]

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
        control_points = self.data['control_points']

        m.CA[0].fix(self.data['CA0'])
        # m.CB[0].fix(self.data['CB0'])
        m.A1.value = self.data['A1']
        # m.A1.fix(self.data['A1'])

        m.k1.pprint()

        m.t_control = control_points

        print('SIMULATING MODEL.')

        # TODO: add simulation for initialization?????
        # Call the simulator (optional)
        sim = Simulator(m, package='casadi')
        tsim, profiles = sim.simulate(integrator='idas')

        print('SIMULATION COMPLETE.')

        # Discretizing the model
        discr = pyo.TransformationFactory("dae.collocation")
        discr.apply_to(m, nfe=self.nfe, ncp=self.ncp, wrt=m.t)

        # sim.initialize_model()


    def label_experiment_impl(self, index_sets_meas):
        """
        Example for annotating (labeling) the model with a
        full experiment.

        Arguments
        ---------

        """
        m = self.model

        # Grab measurement labels
        base_comp_meas = [m.CA, m.CB, ]
        m.experiment_outputs = pyo.Suffix(
            direction=pyo.Suffix.LOCAL,
        )
        m.experiment_outputs.update((k, None) for k in expand_model_components(m, base_comp_meas, index_sets_meas))
        
        # Adding no error for measurements currently
        m.measurement_error = pyo.Suffix(
            direction=pyo.Suffix.LOCAL,
        )
        m.measurement_error.update((k, 1e-2) for k in expand_model_components(m, base_comp_meas, index_sets_meas))

        # Grab design variables
        base_comp_des = [m.CA, ]
        index_sets_des = [[[m.t.first()]], ]
        m.experiment_inputs = pyo.Suffix(
            direction=pyo.Suffix.LOCAL,
        )
        m.experiment_inputs.update((k, pyo.ComponentUID(k)) for k in expand_model_components(m, base_comp_des, index_sets_des))
        
        m.unknown_parameters = pyo.Suffix(
            direction=pyo.Suffix.LOCAL,
        )
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.A1,])

f = open('result.json')
data_ex = json.load(f)
data_ex['control_points'] = {float(k): v for k, v in data_ex['control_points'].items()}

experiments_simple = [
    SimpleReactorExperiment(data_ex, 32, 3),
]

expanded_experiments_simple = [e.get_labeled_model() for e in experiments_simple]