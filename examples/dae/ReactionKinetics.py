#!/usr/bin/env python
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# This example illustrates a simple toolbox for developing chemical
# reaction models based on "mass action kinetics".
#

from pyomo.environ import *
from pyomo.dae import *

from six import itervalues, iteritems, string_types

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

fdiff = TransformationFactory('dae.finite_difference')
colloc = TransformationFactory('dae.collocation')
solver = SolverFactory('ipopt')

class Reaction(object):
    """ A simple class to hold the stoichiometry of a single reaction

    Reaction data is stored in two dictionaries:
       reactants: a map of reactant species name -> stoichiometric coefficient
       products: a map of product species name -> stoichiometric coefficient
    """
    def __init__(self, name, reactants, products=None):
        """ Define a reaction.  The reaction can be specified either as
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
        if isinstance(_in, string_types):
            _in = _in.split('+')
        for x in _in:
            coef, species = self._parseTerm(x)
            ans[species] = ans.get(species,0) + coef
        return ans
        
    def _parseTerm(self, x):
        if isinstance(x, string_types):
            if '*' in x:
                coef, species = x.split('*',1)
                coef = float(coef)
            else:
                coef, species = 1, x
        else:
            coef, species = x
            coef = float(coef)
        return coef, species.strip()

class ReactionNetwork(object):
    """ A simple object to hold sets of reactions. """
    def __init__(self):
        self.reactions = {}

    def add(self, rxn):
        """ Add a single reaction to the reaction network. """
        if rxn.name in self.reactions:
            raise RuntimeError("Duplicate reaction %s:\n\told=%s\n\tnew=%s" %
                               rxn.name, self.reactions[rxn.name], rxn)
        self.reactions[rxn.name] = rxn

    def add_reversible(self, rxn):
        """ Add a pair of reactions to the reaction network.

        This model implements reversible reactions through an explicit
        pair of forward and reverse reactions.
        """
        self.add(rxn)
        tmp = Reaction( name= rxn.name+'_r', 
                        reactants= [(b,a) for a,b in iteritems(rxn.products)],
                        products= [(b,a) for a,b in iteritems(rxn.reactants)] )
        self.add(tmp)

    def species(self):
        """Return the set of all species appearing int he Reaction Network"""
        ans = set()
        for rxn in itervalues(self.reactions):
            ans.update( rxn.reactants )
            ans.update( rxn.products )
        return sorted(ans)


def create_kinetic_model(rxnNet, time):
    """Generate the kinetic model for the specified reaction network on
    the time interval of [0, max(time)].  If time is a list, all time
    points in the list will be used as finite element boundaries."""
    model = ConcreteModel()

    model.rxnNetwork = rxnNet

    model.SPECIES = Set( initialize=rxnNet.species() )
    model.REACTIONS = Set( initialize=rxnNet.reactions.keys() )
    try:
        maxTime = max(time)
        times = time
    except TypeError:
        maxTime = time
        times = [time]
    model.TIME = ContinuousSet( bounds=(0,maxTime), initialize=times )

    model.c = Var( model.TIME, model.SPECIES, bounds=(0,None) )
    model.dcdt = DerivativeVar( model.c, wrt=model.TIME )

    model.k = Var( model.REACTIONS, bounds=(0,None) )
    model.rate = Var( model.TIME, model.REACTIONS )

    def reaction_rate(m, t, r):
        rhs = m.k[r]
        for s, coef in iteritems(m.rxnNetwork.reactions[r].reactants):
            rhs *= m.c[t,s]**coef
        return m.rate[t,r] == rhs
    model.reaction_rate = Constraint( model.TIME, model.REACTIONS,
                                      rule=reaction_rate )

    def stoichiometry(m, t, s):
        rhs = 0
        for r in m.REACTIONS:
            if s in m.rxnNetwork.reactions[r].reactants:
                rhs -= m.rate[t,r] * m.rxnNetwork.reactions[r].reactants[s]
            if s in m.rxnNetwork.reactions[r].products:
                rhs += m.rate[t,r] * m.rxnNetwork.reactions[r].products[s]
        return m.dcdt[t,s] == rhs
    model.stoichiometry = Constraint( model.TIME, model.SPECIES, 
                                      rule=stoichiometry )

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
    rxns.add( Reaction("AtoB", "2*A -> B") )
    rxns.add( Reaction("BtoC", "B -> C") )

    model = create_kinetic_model(rxns, 60*60)

    A1  = 1.32e19 # L / mol*s
    A2  = 1.09e13 # 1/s
    Ea1 = 140000  # J/mol
    Ea2 = 100000  # J/mol
    R   =  8.314  # J / K*mol
    T   =    330  # K

    model.k['AtoB'].fix( A1 * exp( -Ea1 / (R*T) ) )
    model.k['BtoC'].fix( A2 * exp( -Ea2 / (R*T) ) )

    model.c[0, 'A'].fix(1)
    model.c[0, 'B'].fix(0)
    model.c[0, 'C'].fix(0)

    fdiff.apply_to(model, nfe=100)

    results = solver.solve(model, tee=True)

    if plt is not None:
        _tmp = sorted(iteritems(model.c))
        for _i, _x in enumerate('ABC'):
            plt.plot([x[0][0] for x in _tmp if x[0][1] == _x], 
                     [value(x[1]) for x in _tmp if x[0][1] == _x], 
                     'bgr'[_i]+'*', label=_x)
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
    rxns.add( Reaction("AtoB", "2*A -> B") )
    rxns.add( Reaction("BtoC", "B -> C") )

    model = create_kinetic_model(rxns, 60*60)

    A1  = 1.32e19 # L / mol*s
    A2  = 1.09e13 # 1/s
    Ea1 = 140000  # J/mol
    Ea2 = 100000  # J/mol
    R   =  8.314  # J / K*mol
    model.T   =    Var(bounds=(0,None), initialize=330)  # K

    def compute_k(m):
        yield m.k['AtoB'] == A1 * exp( -Ea1 / (R*m.T) )
        yield m.k['BtoC'] == A2 * exp( -Ea2 / (R*m.T) )
    model.compute_k = ConstraintList(rule=compute_k)

    # initial conditions
    model.c[0, 'A'].fix(1)
    model.c[0, 'B'].fix(0)
    model.c[0, 'C'].fix(0)

    fdiff.apply_to(model, nfe=100)

    model.obj = Objective( sense=maximize, 
                           expr=model.c[max(model.TIME), 'B'])

    results = solver.solve(model, tee=True)

    if plt is not None:
        for _i, _x in enumerate('ABC'):
            plt.plot([x.index()[0] for x in model.c[:,_x]], 
                     [value(x) for x in model.c[:,_x]], 
                     'bgr'[_i]+'*', label=_x)
        plt.legend()
        plt.show()

def create_regression_model(b, t):
    rxns = ReactionNetwork()
    rxns.add_reversible( Reaction( "k_1", "TG + MeOH -> DG + FAME" ) )
    rxns.add_reversible( Reaction( "k_2", "DG + MeOH -> MG + FAME" ) )
    rxns.add_reversible( Reaction( "k_3", "MG + MeOH -> Glycerol + FAME" ) )

    data = b.model().data[t]
    key = b.model().key

    model = create_kinetic_model(rxns, data.keys())

    model.T = Param(initialize=t)
    model.error = Var(bounds=(0,None))

    model.compute_error = Constraint(
        expr = model.error == sum(
            (( model.c[t,key[i]] - x ) / max(data[_t][i] for _t in data) )**2 
            for t in data for i,x in enumerate(data[t]) ) )

    return model

def regression_model():
    """ Develop a simple parameter estimation model to identify either
    rate coedfficients (if regress_Ea is False), or the activation
    energy (if regress_Ea is True)."""

    # Model & data from:
    #
    # http://www.doiserbia.nb.rs/img/doi/0367-598X/2014/0367-598X1300037A.pdf
    #
    model = ConcreteModel()

    model.key = key = ('MeOH','TG','DG','MG','FAME','Glycerol')
    model.data = data = {
        150: {
            0:    (2.833,6.84E-02,0.00,0.00,0.00,0.00,),
            256:  (2.807,4.75E-02,1.51E-02,3.71E-03,2.60E-02,8.18E-04,),
            613:  (2.795,3.92E-02,1.98E-02,5.83E-03,3.83E-02,1.60E-03,),
            1228: (2.772,2.95E-02,2.83E-02,9.78E-03,6.07E-02,2.30E-03,),
            1433: (2.762,2.40E-02,3.13E-02,1.49E-02,7.08E-02,4.48E-03,),
            1633: (2.747,1.74E-02,2.02E-02,2.16E-02,8.57E-02,6.23E-03,),
            1933: (2.715,1.03E-02,9.10E-03,2.83E-02,1.18E-01,6.97E-03,),
            2623: (2.699,7.49E-03,7.87E-03,2.34E-02,1.34E-01,9.83E-03,),
            3028: (2.676,3.04E-03,6.56E-03,1.58E-02,1.57E-01,1.68E-02,),
            9000: (2.639,0.00,0.00,0.00,1.94E-01,6.06E-02,), },
        210: {
            0:  (2.835,6.78E-02,0.00,0.00,0.00,0.00,),
            130:(2.806,3.56E-02,1.96E-02,1.92E-02,3.35E-02,0.00,),
            160:(2.755,3.42E-02,1.49E-02,2.54E-02,4.17E-02,0.00,),
            190:(2.735,2.92E-02,1.38E-02,2.83E-02,5.67E-02,0.00,),
            220:(2.715,2.20E-02,1.40E-02,2.80E-02,7.97E-02,4.37E-03,),
            250:(2.698,1.70E-02,7.89E-03,3.12E-02,1.05E-01,1.24E-02,),
            280:(2.675,1.29E-02,5.45E-03,2.78E-02,1.28E-01,2.23E-02,),
            340:(2.659,7.02E-03,5.90E-03,1.56E-02,1.58E-01,3.99E-02,),
            400:(2.648,3.65E-03,5.13E-03,7.92E-03,1.75E-01,5.17E-02,),
            460:(2.641,2.66E-03,5.04E-03,4.64E-03,1.79E-01,5.61E-02,),
            520:(2.637,1.49E-03,3.57E-03,2.48E-03,1.86E-01,6.09E-02,),
            580:(2.633,3.35E-04,4.96E-04,1.84E-03,1.95E-01,6.58E-02,),
            640:(2.632,2.49E-04,2.40E-04,1.44E-03,1.98E-01,6.65E-02,),
            700:(2.630,2.31E-04,2.90E-05,1.28E-03,2.00E-01,6.69E-02,),
            760:(2.630,0.00,0.00,7.61E-04,2.02E-01,6.77E-02,), }
        }

    model.experiment = Block( data.keys(), rule=create_regression_model )

    model.obj = Objective( sense=minimize, 
                           expr=sum(b.error for b in model.experiment[:]) )
    _experiments = list( model.experiment.values() )

    # initializations from the paper
    for _e in _experiments:
        _e.k['k_1']   = 7.58e-7
        _e.k['k_1_r'] = 0
        _e.k['k_2']   = 2.20e-7
        _e.k['k_2_r'] = 0
        _e.k['k_3']   = 2.15e-7
        _e.k['k_3_r'] = 0

    #fdiff.apply_to(model, nfe=100)
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
        model.Kset = Set(initialize=['k_1','k_2','k_3',])
        model.Ea = Var(model.Kset, bounds=(0,None), initialize=0)
        model.A = Var(model.Kset, bounds=(0,None), initialize=0)
        model.R = Param(initialize=8.314)
        for _e in _experiments:
            _e.k.fix()
            def compute_k(e, _k):
                m = e.model()
                # k11' == k_mt + k_11 * (C_DG + C_MG) / C_TG_0
                #return e.k[_k] == m.A[_k] * exp( -m.Ea[_k] / ( m.R * e.T ) )
                return log(e.k[_k]) == log(m.A[_k]) - m.Ea[_k] / ( m.R * e.T )
            _e.compute_k = Constraint(model.Kset, rule=compute_k)
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
                _ax.plot( [ t for t in data[T].keys() ],
                          [ data[T][t][_i] for t in data[T].keys() ],
                          'mkrgbc'[_i]+'x' )
            for _i, _x in enumerate(key):
                _ax = ax2 if _x == 'MeOH' else ax
                _ax.plot([ x.index()[0] for x in model.experiment[T].c[:,_x] ],
                         [ value(x) for x in model.experiment[T].c[:,_x] ],
                         'mkrgbc'[_i]+'-')
        plt.show()
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in '123':
        print("""ERROR: expected a model to run:
    1 - simple simulation model
    2 - simple (final value) optimization model
    3 - kinetic parameter regression model""")
        sys.exit(1)

    if '1' in sys.argv[1]:
        simple_simulation_model()
    if '2' in sys.argv[1]:
        simple_optimization_model()
    if '3' in sys.argv[1]:
        regression_model()
