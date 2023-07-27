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
        from pyomo.contrib.edi import Formulation
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
        import pyomo
        from pyomo.contrib.edi import Formulation
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

        x12 = f.Variable(name = 'x12', guess = 1.0, units = 'm'  , description = 'The x variable', size = None, bounds=[-10,10], domain=None )
        self.assertRaises(ValueError, f.Variable, **{'name':'x13', 'guess':1.0 , 'units' :'m'  , 'description': 'The x variable', 'size': None, 'bounds':[10,-10], 'domain':None })
        self.assertRaises(ValueError, f.Variable, **{'name':'x14', 'guess':1.0 , 'units' :'m'  , 'description': 'The x variable', 'size': None, 'bounds':["-10","10"], 'domain':None })
        self.assertRaises(ValueError, f.Variable, **{'name':'x15', 'guess':1.0 , 'units' :'m'  , 'description': 'The x variable', 'size': None, 'bounds':[1,2,3], 'domain':None })
        self.assertRaises(ValueError, f.Variable, **{'name':'x16', 'guess':1.0 , 'units' :'m'  , 'description': 'The x variable', 'size': None, 'bounds':"error", 'domain':None })
        self.assertRaises(ValueError, f.Variable, **{'name':'x17', 'guess':1.0 , 'units' :'m'  , 'description': 'The x variable', 'size': None, 'bounds':[0,"10"], 'domain':None })


    def test_edi_formulation_constant(self):
        "Tests the constant constructor in edi.formulation"
        from pyomo.contrib.edi import Formulation
        from pyomo.environ import Reals, PositiveReals

        f = Formulation()

        c1 = f.Constant(name = 'c1', value = 1.0, units = 'm', description = 'A constant c', size = None, within = None)
        self.assertRaises(RuntimeError,f.Constant,*('c1', 1.0, 'm'))
        c2 = f.Constant('c2', 1.0, 'm')
        c3 = f.Constant('c3', 1.0, 'm', 'A constant c', None, None)
        c4 = f.Constant(name = 'c4', value = 1.0, units = 'm'  , description = 'A constant c', size = None, within=PositiveReals )
        self.assertRaises(RuntimeError, f.Constant, **{'name':'c5', 'value':1.0 , 'units' :'m'  , 'description': 'The x variable', 'size': None, 'within':"error" })

        c6 = f.Constant(name = 'c6', value = 1.0, units = 'm'  , description = 'A constant c', size = 0, within=None )
        c7 = f.Constant(name = 'c7', value = 1.0, units = 'm'  , description = 'A constant c', size = 5, within=None )
        self.assertRaises(ValueError, f.Constant, **{'name':'c8', 'value':1.0 , 'units' :'m'  , 'description': 'A constant c', 'size': 'error', 'within':None })
        c9 = f.Constant(name = 'c9', value = 1.0, units = 'm'  , description = 'A constant c', size = [2,2], within=None )
        self.assertRaises(ValueError, f.Constant, **{'name':'c10', 'value':1.0 , 'units' :'m'  , 'description': 'A constant c', 'size': ['2','2'], 'within':None })
        self.assertRaises(ValueError, f.Constant, **{'name':'c11', 'value':1.0 , 'units' :'m'  , 'description': 'A constant c', 'size': [2,1], 'within':None })


    def test_edi_formulation_objective(self):
        "Tests the objective constructor in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
        y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')
        f.Objective( x + y )

    def test_edi_formulation_runtimeobjective(self):
        "Tests the runtime objective constructor in edi.formulation"
        # TODO: not currently implemented, see:  https://github.com/codykarcher/pyomo/issues/5
        pass

    def test_edi_formulation_constraint(self):
        "Tests the constraint constructor in edi.formulation"
        pass    


    def test_edi_formulation_runtimeconstraint(self):
        "Tests the runtime constraint constructor in edi.formulation"
        pass    


    def test_edi_formulation_constraintlist(self):
        "Tests the constraint list constructor in edi.formulation"
        pass    

    def test_edi_formulation_getvariables(self):
        "Tests the get_variables function in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation
        f = Formulation()
        x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
        y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')

        vrs = f.get_variables()
        self.assertListEqual(vrs, [x,y])

    def test_edi_formulation_getconstants(self):
        "Tests the get_constants function in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation
        f = Formulation()
        x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
        y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')

        c1 = f.Constant(name = 'c1', value = 1.0, units = 'm', description = 'A constant c1', size = None, within = None)
        c2 = f.Constant(name = 'c2', value = 1.0, units = 'm', description = 'A constant c2', size = None, within = None)

        csts = f.get_constants()
        self.assertListEqual(csts, [c1,c2])

    def test_edi_formulation_getobjectives(self):
        "Tests the get_objectives function in edi.formulation"
        pass    

    def test_edi_formulation_getconstraints(self):
        "Tests the get_constraints function in edi.formulation"
        pass    

    def test_edi_formulation_getexplicitconstraints(self):
        "Tests the get_explicitConstraints function in edi.formulation"
        pass    

    def test_edi_formulation_getruntimeconstraints(self):
        "Tests the get_runtimeConstraints function in edi.formulation"
        pass    

    def test_edi_formulation_checkunits(self):
        "Tests the check_units function in edi.formulation"
        pass    


if __name__ == '__main__':
    unittest.main()

















