#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet
from pyomo.contrib.matching.interface import IncidenceGraphInterface
import pyomo.common.unittest as unittest

def make_gas_expansion_model(N=2):
    """
    This is the simplest model I could think of that has a
    subsystem with a non-trivial block triangularization.
    Something like a gas (somehow) undergoing a series
    of isentropic expansions.
    """
    m = pyo.ConcreteModel()
    m.streams = pyo.Set(initialize=range(N+1))
    m.rho = pyo.Var(m.streams, initialize=1)
    m.P = pyo.Var(m.streams, initialize=1)
    m.F = pyo.Var(m.streams, initialize=1)
    m.T = pyo.Var(m.streams, initialize=1)

    m.R = pyo.Param(initialize=8.31)
    m.Q = pyo.Param(m.streams, initialize=1)
    m.gamma = pyo.Param(initialize=1.4*m.R.value)

    def mbal(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        else:
            return m.rho[i]*m.F[i] == m.rho[i-1]*m.F[i-1]
    m.mbal = pyo.Constraint(m.streams, rule=mbal)

    def ebal(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        else:
            return (
                    m.rho[i-1]*m.F[i-1]*m.T[i-1] +
                    m.Q[i] ==
                    m.rho[i]*m.F[i]*m.T[i]
                    )
    m.ebal = pyo.Constraint(m.streams, rule=ebal)

    def expansion(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        else:
            return m.P[i]/m.P[i-1] == (m.rho[i]/m.rho[i-1])**m.gamma
    m.expansion = pyo.Constraint(m.streams, rule=expansion)

    def ideal_gas(m, i):
        return m.P[i] == m.rho[i]*m.R*m.T[i]
    m.ideal_gas = pyo.Constraint(m.streams, rule=ideal_gas)

    return m


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
class TestGasExpansionModel(unittest.TestCase):
    def test_imperfect_matching(self):
        model = make_gas_expansion_model()
        model.obj = pyo.Objective(expr=0)
        igraph = IncidenceGraphInterface(model)

        n_eqn = len(list(model.component_data_objects(pyo.Constraint)))
        matching = igraph.maximum_matching()
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_eqn)
        self.assertEqual(len(values), n_eqn)

    def test_perfect_matching(self):
        model = make_gas_expansion_model()
        model.obj = pyo.Objective(expr=0)
        igraph = IncidenceGraphInterface(model)

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        n_var = len(variables)
        matching = igraph.maximum_matching(variables, constraints)
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_var)
        self.assertEqual(len(values), n_var)

        # The subset of variables and equations we have identified
        # do not have a unique perfect matching. But we at least know
        # this much.
        self.assertIs(matching[model.ideal_gas[0]], model.P[0])

    def test_triangularize(self):
        N = 5
        model = make_gas_expansion_model(N)
        model.obj = pyo.Objective(expr=0)
        igraph = IncidenceGraphInterface(model)

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        var_block_map, con_block_map = igraph.block_triangularize(
                variables, constraints)
        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N+1)
        self.assertEqual(len(con_values), N+1)

        self.assertEqual(var_block_map[model.P[0]], 0)

        for i in model.streams:
            if i != model.streams.first():
                self.assertEqual(var_block_map[model.rho[i]], i)
                self.assertEqual(var_block_map[model.T[i]], i)
                self.assertEqual(var_block_map[model.P[i]], i)
                self.assertEqual(var_block_map[model.F[i]], i)

                self.assertEqual(con_block_map[model.ideal_gas[i]], i)
                self.assertEqual(con_block_map[model.expansion[i]], i)
                self.assertEqual(con_block_map[model.mbal[i]], i)
                self.assertEqual(con_block_map[model.ebal[i]], i)

    def test_exception(self):
        model = make_gas_expansion_model()
        model.obj = pyo.Objective(expr=0)
        igraph = IncidenceGraphInterface(model)

        with self.assertRaises(ValueError) as exc:
            variables = [model.P]
            constraints = [model.ideal_gas]
            igraph.maximum_matching(variables, constraints)
        self.assertIn('must be unindexed', str(exc.exception))

        with self.assertRaises(ValueError) as exc:
            variables = [model.P]
            constraints = [model.ideal_gas]
            igraph.block_triangularize(variables, constraints)
        self.assertIn('must be unindexed', str(exc.exception))

if __name__ == "__main__":
    unittest.main()
