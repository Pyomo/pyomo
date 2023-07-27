#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
# from pyomo.opt import check_optimal_termination
from pyomo.common.dependencies import attempt_import

# np, numpy_available = attempt_import(
#     'numpy', 'edi requires numpy'
# )
# scipy, scipy_available = attempt_import(
#     'scipy', 'inverse_reduced_hessian requires scipy'
# )

# if not (numpy_available and scipy_available):
# if not numpy_available:
#     raise unittest.SkipTest(
#         'edi.formulation tests require numpy'
#     )

class TestEDIFormulation(unittest.TestCase):
    def test_edi_formulation_init(self):
        "Tests that a formulation initializes to the correct type and has proper data"
        from pyomo.environ import ConcreteModel
        from pyomo.contrib.edi.formulation import Formulation
        f = Formulation()

        self.assertIsInstance(f, Formulation)
        self.assertIsInstance(f, ConcreteModel)

        self.assertEqual(f._objective_counter      , 0  )
        self.assertEqual(f._constraint_counter     , 0  )
        self.assertEqual(f._variable_keys          , [] )
        self.assertEqual(f._constant_keys          , [] )
        self.assertEqual(f._objective_keys         , [] )
        self.assertEqual(f._runtimeObjective_keys  , [] )
        self.assertEqual(f._objective_keys         , [] )
        self.assertEqual(f._runtimeConstraint_keys , [] )
        self.assertEqual(f._constraint_keys        , [] )
        self.assertEqual(f._allConstraint_keys     , [] )


    def test_edi_formulation_variable(self):
        "Tests the variable constructor in edi.formulation"
        from pyomo.contrib.edi.formulation import Formulation
        from pyomo.environ import Reals, PositiveReals

        f = Formulation()

        x1 = f.Variable(name = 'x1', guess = 1.0, units = 'm'  , description = 'The x variable', size = None, bounds=None, domain=None )
        self.assertRaises(RuntimeError,f.Variable,*('x1', 1.0, 'm'))
        x2 = f.Variable('x2', 1.0, 'm')
        x3 = f.Variable('x3', 1.0, 'm' , 'The x variable', None, None, None )
        x4 = f.Variable(name = 'x4', guess = 1.0, units = 'm'  , description = 'The x variable', size = None, bounds=None, domain=PositiveReals )
        self.assertRaises(RuntimeError, f.Variable, **{'name':'x5', 'guess':1.0 , 'units' :'m'  , 'description': 'The x variable', 'size': None, 'bounds':None, 'domain':"error" })

        x6 = f.Variable(name = 'x6', guess = 1.0, units = 'm'  , description = 'The x variable', size = 0, bounds=None, domain=None )
        x7 = f.Variable(name = 'x7', guess = 1.0, units = 'm'  , description = 'The x variable', size = 5, bounds=None, domain=None )
        self.assertRaises(ValueError, f.Variable, **{'name':'x8', 'guess':1.0 , 'units' :'m'  , 'description': 'The x variable', 'size': 'error', 'bounds':None, 'domain':None })

        x9 = f.Variable(name = 'x9', guess = 1.0, units = 'm'  , description = 'The x variable', size = [2,2], bounds=None, domain=None )
        self.assertRaises(ValueError, f.Variable, **{'name':'x10', 'guess':1.0 , 'units' :'m'  , 'description': 'The x variable', 'size': ['2','2'], 'bounds':None, 'domain':None })
        self.assertRaises(ValueError, f.Variable, **{'name':'x11', 'guess':1.0 , 'units' :'m'  , 'description': 'The x variable', 'size': [2,1], 'bounds':None, 'domain':None })





# def Variable(self, name, guess, units, description='', size = None, bounds=None, domain=None):
# def Constant(self, name, value, units, description, size=None, within=None):
# def Objective(self, expr, sense=minimize):
# def RuntimeObjective(self):
# def Constraint(self, expr):
# def CoverConstr_rule(model, i): return a[i] * model.y[i] >= b[i]
# def RuntimeConstraint(self, rcCon):
# def ConstraintList(self, conList):
# def get_variables(self):
# def get_constants(self):
# def get_objectives(self):
# def get_constraints(self):
# def get_explicitConstraints(self):
# def get_runtimeConstraints(self):
# def solve(self):

if __name__ == '__main__':
    unittest.main()

















