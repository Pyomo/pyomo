#!/usr/bin/env python
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

#
# This example illustrates a simple toolbox for developing chemical
# reaction models based on "mass action kinetics".
#

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

fdiff = pyo.TransformationFactory('dae.finite_difference')
colloc = pyo.TransformationFactory('dae.collocation')
solver = pyo.SolverFactory('ipopt')


class Reaction(object):
    """A simple class to hold the stoichiometry of a single reaction

    Reaction data is stored in two dictionaries:
       reactants: a map of reactant species name -> stoichiometric coefficient
       products: a map of product species name -> stoichiometric coefficient
    """

    def __init__(self, name, reactants, products=None):
        """Define a reaction.  The reaction can be specified either as
        a text string:

            Reaction("2*A + B -> C + 3*D")

        or reactant and product strings:

            Reaction(reactants="2*A + B", products="C + 3*D")

        or lists of terms reactant and product strings:

            Reaction(reactants=["2*A","B"], products=["C","3*D"])

        or lists of (coef, name) tuples:

            Reaction(reactants=[(2,"A"),(1,"B")], products=[(1,"C"),(3,"D")])

        or any (sensible) combination of the above.
        """
        if products is None:
            reactants, products = reactants.split('->')
        self.name = name
        self.reactants = self._parse(reactants)
        self.products = self._parse(products)

    def _parse(self, _in):
        ans = {}
        if isinstance(_in, str):
            _in = _in.split('+')
        for x in _in:
            coef, species = self._parseTerm(x)
            ans[species] = ans.get(species, 0) + coef
        return ans

    def _parseTerm(self, x):
        if isinstance(x, str):
            if '*' in x:
                coef, species = x.split('*', 1)
                coef = float(coef)
            else:
                coef, species = 1, x
        else:
            coef, species = x
            coef = float(coef)
        return coef, species.strip()


class ReactionNetwork(object):
    """A simple object to hold sets of reactions."""

    def __init__(self):
        self.reactions = {}

    def add(self, rxn):
        """Add a single reaction to the reaction network."""
        if rxn.name in self.reactions:
            raise RuntimeError(
                "Duplicate reaction %s:\n\told=%s\n\tnew=%s" % rxn.name,
                self.reactions[rxn.name],
                rxn,
            )
        self.reactions[rxn.name] = rxn

    def add_reversible(self, rxn):
        """Add a pair of reactions to the reaction network.

        This model implements reversible reactions through an explicit
        pair of forward and reverse reactions.
        """
        self.add(rxn)
        tmp = Reaction(
            name=rxn.name + '_r',
            reactants=[(b, a) for a, b in rxn.products.items()],
            products=[(b, a) for a, b in rxn.reactants.items()],
        )
        self.add(tmp)

    def species(self):
        """Return the set of all species appearing int he Reaction Network"""
        ans = set()
        for rxn in self.reactions.values():
            ans.update(rxn.reactants)
            ans.update(rxn.products)
        return sorted(ans)


def create_kinetic_model(rxnNet, time):
    """Generate the kinetic model for the specified reaction network on
    the time interval of [0, max(time)].  If time is a list, all time
    points in the list will be used as finite element boundaries."""
    model = pyo.ConcreteModel()

    model.rxnNetwork = rxnNet

    model.SPECIES = pyo.Set(initialize=rxnNet.species())
    model.REACTIONS = pyo.Set(initialize=rxnNet.reactions.keys())
    try:
        maxTime = max(time)
        times = time
    except TypeError:
        maxTime = time
        times = [time]
    model.TIME = ContinuousSet(bounds=(0, maxTime), initialize=times)

    model.c = pyo.Var(model.TIME, model.SPECIES, bounds=(0, None))
    model.dcdt = DerivativeVar(model.c, wrt=model.TIME)

    model.k = pyo.Var(model.REACTIONS, bounds=(0, None))
    model.rate = pyo.Var(model.TIME, model.REACTIONS)

    def reaction_rate(m, t, r):
        rhs = m.k[r]
        for s, coef in m.rxnNetwork.reactions[r].reactants.items():
            rhs *= m.c[t, s] ** coef
        return m.rate[t, r] == rhs

    model.reaction_rate = pyo.Constraint(
        model.TIME, model.REACTIONS, rule=reaction_rate
    )

    def stoichiometry(m, t, s):
        rhs = 0
        for r in m.REACTIONS:
            if s in m.rxnNetwork.reactions[r].reactants:
                rhs -= m.rate[t, r] * m.rxnNetwork.reactions[r].reactants[s]
            if s in m.rxnNetwork.reactions[r].products:
                rhs += m.rate[t, r] * m.rxnNetwork.reactions[r].products[s]
        return m.dcdt[t, s] == rhs

    model.stoichiometry = pyo.Constraint(model.TIME, model.SPECIES, rule=stoichiometry)

    return model


#
# This example is based on
#
#   http://www.comsol.com/blogs/general-introduction-chemical-kinetics-arrhenius-law/
#
# with some slightly adjusted coefficients (Ea2 & T): I could not match
# the results from the example using the coefficients that they give
#
def simple_simulation_model():
    """Run a simple simulation model for 2*A -> B -> C."""

    rxns = ReactionNetwork()
    rxns.add(Reaction("AtoB", "2*A -> B"))
    rxns.add(Reaction("BtoC", "B -> C"))

    model = create_kinetic_model(rxns, 60 * 60)

    A1 = 1.32e19  # L / mol*s
    A2 = 1.09e13  # 1/s
    Ea1 = 140000  # J/mol
    Ea2 = 100000  # J/mol
    R = 8.314  # J / K*mol
    T = 330  # K

    model.k['AtoB'].fix(A1 * pyo.exp(-Ea1 / (R * T)))
    model.k['BtoC'].fix(A2 * pyo.exp(-Ea2 / (R * T)))

    model.c[0, 'A'].fix(1)
    model.c[0, 'B'].fix(0)
    model.c[0, 'C'].fix(0)

    fdiff.apply_to(model, nfe=100)

    results = solver.solve(model, tee=True)

    if plt is not None:
        _tmp = sorted(model.c.items())
        for _i, _x in enumerate('ABC'):
            plt.plot(
                [x[0][0] for x in _tmp if x[0][1] == _x],
                [pyo.value(x[1]) for x in _tmp if x[0][1] == _x],
                'bgr'[_i] + '*',
                label=_x,
            )
        plt.legend()
        plt.show()


#
# This example is based on
#
#   http://www.comsol.com/blogs/general-introduction-chemical-kinetics-arrhenius-law/
#
# with some slightly adjusted coefficients (Ea2 & T): I could not match
# the results from the example using the coefficients that they give
#
def simple_optimization_model():
    """Optimize the temperature in order to maximize the final
    concentration of the intermediate, "B"."""

    rxns = ReactionNetwork()
    rxns.add(Reaction("AtoB", "2*A -> B"))
    rxns.add(Reaction("BtoC", "B -> C"))

    model = create_kinetic_model(rxns, 60 * 60)

    A1 = 1.32e19  # L / mol*s
    A2 = 1.09e13  # 1/s
    Ea1 = 140000  # J/mol
    Ea2 = 100000  # J/mol
    R = 8.314  # J / K*mol
    model.T = pyo.Var(bounds=(0, None), initialize=330)  # K

    def compute_k(m):
        yield m.k['AtoB'] == A1 * pyo.exp(-Ea1 / (R * m.T))
        yield m.k['BtoC'] == A2 * pyo.exp(-Ea2 / (R * m.T))

    model.compute_k = pyo.ConstraintList(rule=compute_k)

    # initial conditions
    model.c[0, 'A'].fix(1)
    model.c[0, 'B'].fix(0)
    model.c[0, 'C'].fix(0)

    fdiff.apply_to(model, nfe=100)

    model.obj = pyo.Objective(sense=pyo.maximize, expr=model.c[max(model.TIME), 'B'])

    results = solver.solve(model, tee=True)

    if plt is not None:
        for _i, _x in enumerate('ABC'):
            plt.plot(
                [x.index()[0] for x in model.c[:, _x]],
                [pyo.value(x) for x in model.c[:, _x]],
                'bgr'[_i] + '*',
                label=_x,
            )
        plt.legend()
        plt.show()


def create_regression_model(b, t):
    rxns = ReactionNetwork()
    rxns.add_reversible(Reaction("k_1", "TG + MeOH -> DG + FAME"))
    rxns.add_reversible(Reaction("k_2", "DG + MeOH -> MG + FAME"))
    rxns.add_reversible(Reaction("k_3", "MG + MeOH -> Glycerol + FAME"))

    data = b.model().data[t]
    key = b.model().key

    model = create_kinetic_model(rxns, data.keys())

    model.T = pyo.Param(initialize=t)
    model.error = pyo.Var(bounds=(0, None))

    model.compute_error = pyo.Constraint(
        expr=model.error
        == sum(
            ((model.c[t, key[i]] - x) / max(data[_t][i] for _t in data)) ** 2
            for t in data
            for i, x in enumerate(data[t])
        )
    )

    return model


def regression_model():
    """Develop a simple parameter estimation model to identify either
    rate coedfficients (if regress_Ea is False), or the activation
    energy (if regress_Ea is True)."""

    # Model & data from:
    #
    # Almagrbi, A. M., Hatami, T., Glišić, S., & Orlović,
    # A. Determination of kinetic parameters for complex
    # transesterification reaction by standard optimisation methods.
    # Hem. Ind. 68(2) 149–159 (2014).  doi:10.2298/HEMIND130118037A
    #
    model = pyo.ConcreteModel()

    model.key = key = ('MeOH', 'TG', 'DG', 'MG', 'FAME', 'Glycerol')
    model.data = data = {
        150: {
            0: (2.833, 6.84e-02, 0.00, 0.00, 0.00, 0.00),
            256: (2.807, 4.75e-02, 1.51e-02, 3.71e-03, 2.60e-02, 8.18e-04),
            613: (2.795, 3.92e-02, 1.98e-02, 5.83e-03, 3.83e-02, 1.60e-03),
            1228: (2.772, 2.95e-02, 2.83e-02, 9.78e-03, 6.07e-02, 2.30e-03),
            1433: (2.762, 2.40e-02, 3.13e-02, 1.49e-02, 7.08e-02, 4.48e-03),
            1633: (2.747, 1.74e-02, 2.02e-02, 2.16e-02, 8.57e-02, 6.23e-03),
            1933: (2.715, 1.03e-02, 9.10e-03, 2.83e-02, 1.18e-01, 6.97e-03),
            2623: (2.699, 7.49e-03, 7.87e-03, 2.34e-02, 1.34e-01, 9.83e-03),
            3028: (2.676, 3.04e-03, 6.56e-03, 1.58e-02, 1.57e-01, 1.68e-02),
            9000: (2.639, 0.00, 0.00, 0.00, 1.94e-01, 6.06e-02),
        },
        210: {
            0: (2.835, 6.78e-02, 0.00, 0.00, 0.00, 0.00),
            130: (2.806, 3.56e-02, 1.96e-02, 1.92e-02, 3.35e-02, 0.00),
            160: (2.755, 3.42e-02, 1.49e-02, 2.54e-02, 4.17e-02, 0.00),
            190: (2.735, 2.92e-02, 1.38e-02, 2.83e-02, 5.67e-02, 0.00),
            220: (2.715, 2.20e-02, 1.40e-02, 2.80e-02, 7.97e-02, 4.37e-03),
            250: (2.698, 1.70e-02, 7.89e-03, 3.12e-02, 1.05e-01, 1.24e-02),
            280: (2.675, 1.29e-02, 5.45e-03, 2.78e-02, 1.28e-01, 2.23e-02),
            340: (2.659, 7.02e-03, 5.90e-03, 1.56e-02, 1.58e-01, 3.99e-02),
            400: (2.648, 3.65e-03, 5.13e-03, 7.92e-03, 1.75e-01, 5.17e-02),
            460: (2.641, 2.66e-03, 5.04e-03, 4.64e-03, 1.79e-01, 5.61e-02),
            520: (2.637, 1.49e-03, 3.57e-03, 2.48e-03, 1.86e-01, 6.09e-02),
            580: (2.633, 3.35e-04, 4.96e-04, 1.84e-03, 1.95e-01, 6.58e-02),
            640: (2.632, 2.49e-04, 2.40e-04, 1.44e-03, 1.98e-01, 6.65e-02),
            700: (2.630, 2.31e-04, 2.90e-05, 1.28e-03, 2.00e-01, 6.69e-02),
            760: (2.630, 0.00, 0.00, 7.61e-04, 2.02e-01, 6.77e-02),
        },
    }

    model.experiment = pyo.Block(data.keys(), rule=create_regression_model)

    model.obj = pyo.Objective(
        sense=pyo.minimize, expr=sum(b.error for b in model.experiment[:])
    )
    _experiments = list(model.experiment.values())

    # initializations from the paper
    for _e in _experiments:
        _e.k['k_1'] = 7.58e-7
        _e.k['k_1_r'] = 0
        _e.k['k_2'] = 2.20e-7
        _e.k['k_2_r'] = 0
        _e.k['k_3'] = 2.15e-7
        _e.k['k_3_r'] = 0

    # fdiff.apply_to(model, nfe=100)
    colloc.apply_to(model, nfe=100, ncp=3)

    # Note that the two experiments are not linked at this point, so
    # this solve is effectively performing two independent regressions
    # at the same time.
    results = solver.solve(model, tee=True)

    # regress_Ea will turn in the activation energy estimation by
    # linking the experiments and re-regressing (using the previous
    # independent regression as the starting point)
    regress_Ea = True
    if regress_Ea:
        model.Kset = pyo.Set(initialize=['k_1', 'k_2', 'k_3'])
        model.Ea = pyo.Var(model.Kset, bounds=(0, None), initialize=0)
        model.A = pyo.Var(model.Kset, bounds=(0, None), initialize=0)
        model.R = pyo.Param(initialize=8.314)
        for _e in _experiments:
            _e.k.fix()

            def compute_k(e, _k):
                m = e.model()
                # k11' == k_mt + k_11 * (C_DG + C_MG) / C_TG_0
                # return e.k[_k] == m.A[_k] * pyo.exp( -m.Ea[_k] / ( m.R * e.T ) )
                return pyo.log(e.k[_k]) == pyo.log(m.A[_k]) - m.Ea[_k] / (m.R * e.T)

            _e.compute_k = pyo.Constraint(model.Kset, rule=compute_k)
        solver.solve(model, tee=True)

    for _e in _experiments:
        _e.k.pprint()
    if regress_Ea:
        model.A.pprint()
        model.Ea.pprint()

    if plt is not None:
        for T in data:
            plt.figure()
            ax = plt.gca()
            ax2 = plt.twinx()
            for _i, _x in enumerate(key):
                _ax = ax2 if _x == 'MeOH' else ax
                _ax.plot(
                    [t for t in data[T].keys()],
                    [data[T][t][_i] for t in data[T].keys()],
                    'mkrgbc'[_i] + 'x',
                )
            for _i, _x in enumerate(key):
                _ax = ax2 if _x == 'MeOH' else ax
                _ax.plot(
                    [x.index()[0] for x in model.experiment[T].c[:, _x]],
                    [pyo.value(x) for x in model.experiment[T].c[:, _x]],
                    'mkrgbc'[_i] + '-',
                )
        plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2 or sys.argv[1] not in '123':
        print(
            """ERROR: expected a model to run:
    1 - simple simulation model
    2 - simple (final value) optimization model
    3 - kinetic parameter regression model"""
        )
        sys.exit(1)

    if '1' in sys.argv[1]:
        simple_simulation_model()
    if '2' in sys.argv[1]:
        simple_optimization_model()
    if '3' in sys.argv[1]:
        regression_model()
