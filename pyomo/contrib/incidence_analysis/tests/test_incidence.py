#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
    IncidenceMethod,
    get_incident_variables,
)


class _TestIncidence(object):
    def _get_incident_variables(self, expr):
        raise NotImplementedError(
            "_TestIncidence should not be used directly"
        )

    def test_basic_incidence(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] + m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))

    def test_incidence_with_fixed_variable(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] + m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
        m.x[2].fix()
        variables = self._get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[1], m.x[3]]))

    def test_incidence_with_mutable_parameter(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.p = pyo.Param(mutable=True, initialize=None)
        expr = m.x[1] + m.p * m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))
        

class TestIncidenceStandardRepn(unittest.TestCase, _TestIncidence):
    def _get_incident_variables(self, expr, **kwds):
        method = IncidenceMethod.standard_repn
        return get_incident_variables(expr, method=method, **kwds)

    def test_zero_coef(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])

        # generate_standard_repn filters subexpressions with zero coefficients
        expr = 0 * m.x[1] + 0 * m.x[1] * m.x[2] + 0 * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

    def test_variable_minus_itself(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        # standard repn will recognize the zero coefficient and filter x[1]
        expr = m.x[1] + m.x[2] * m.x[3] - m.x[1]
        variables = self._get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[2], m.x[3]]))

    def test_linear_only(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])

        expr = 2 * m.x[1] + 4 * m.x[2] * m.x[1] - m.x[1] * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr, linear_only=True)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1]]))


class TestIncidenceIdentifyVariables(unittest.TestCase, _TestIncidence):
    def _get_incident_variables(self, expr, **kwds):
        method = IncidenceMethod.identify_variables
        return get_incident_variables(expr, method=method, **kwds)

    def test_zero_coef(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])

        # identify_variables does not eliminate expressions times zero
        expr = 0 * m.x[1] + 0 * m.x[1] * m.x[2] + 0 * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))

    def test_variable_minus_itself(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        # identify_variables will not filter x[1]
        expr = m.x[1] + m.x[2] * m.x[3] - m.x[1]
        variables = self._get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet(m.x[:]))


if __name__ == "__main__":
    unittest.main()
