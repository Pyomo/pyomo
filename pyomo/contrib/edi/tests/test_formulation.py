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
from pyomo.common.dependencies import attempt_import

from pyomo.core.base.units_container import pint_available

from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.dependencies import scipy, scipy_available

egb, egb_available = attempt_import(
    "pyomo.contrib.pynumero.interfaces.external_grey_box"
)

formulation_available = False
try:
    from pyomo.contrib.edi import Formulation

    formulation_available = True
except:
    pass
    # formulation_available = False

blackbox_available = False
try:
    from pyomo.contrib.edi import BlackBoxFunctionModel

    blackbox_available = True
except:
    pass
    # blackbox_available = False

if numpy_available:
    import numpy as np


@unittest.skipIf(
    not egb_available, 'Testing pyomo.contrib.edi requires pynumero external grey boxes'
)
@unittest.skipIf(not formulation_available, 'Formulation import failed')
@unittest.skipIf(not blackbox_available, 'Blackbox import failed')
@unittest.skipIf(not numpy_available, 'Testing pyomo.contrib.edi requires numpy')
@unittest.skipIf(not scipy_available, 'Testing pyomo.contrib.edi requires scipy')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
class TestEDIFormulation(unittest.TestCase):
    def test_edi_formulation_init(self):
        "Tests that a formulation initializes to the correct type and has proper data"
        from pyomo.environ import ConcreteModel
        from pyomo.contrib.edi import Formulation

        f = Formulation()

        self.assertIsInstance(f, Formulation)
        self.assertIsInstance(f, ConcreteModel)

        self.assertEqual(f._objective_counter, 0)
        self.assertEqual(f._constraint_counter, 0)
        self.assertEqual(f._variable_keys, [])
        self.assertEqual(f._constant_keys, [])
        self.assertEqual(f._objective_keys, [])
        self.assertEqual(f._runtimeObjective_keys, [])
        self.assertEqual(f._objective_keys, [])
        self.assertEqual(f._runtimeConstraint_keys, [])
        self.assertEqual(f._constraint_keys, [])
        self.assertEqual(f._allConstraint_keys, [])

    def test_edi_formulation_variable(self):
        "Tests the variable constructor in edi.formulation"
        import pyomo
        from pyomo.contrib.edi import Formulation
        from pyomo.environ import Reals, PositiveReals

        f = Formulation()

        x1 = f.Variable(
            name='x1',
            guess=1.0,
            units='m',
            description='The x variable',
            size=None,
            bounds=None,
            domain=None,
        )
        self.assertRaises(RuntimeError, f.Variable, *('x1', 1.0, 'm'))
        x2 = f.Variable('x2', 1.0, 'm')
        x3 = f.Variable('x3', 1.0, 'm', 'The x variable', None, None, None)
        x4 = f.Variable(
            name='x4',
            guess=1.0,
            units='m',
            description='The x variable',
            size=None,
            bounds=None,
            domain=PositiveReals,
        )
        self.assertRaises(
            RuntimeError,
            f.Variable,
            **{
                'name': 'x5',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'bounds': None,
                'domain': "error",
            }
        )

        x6 = f.Variable(
            name='x6',
            guess=1.0,
            units='m',
            description='The x variable',
            size=0,
            bounds=None,
            domain=None,
        )
        x7 = f.Variable(
            name='x7',
            guess=1.0,
            units='m',
            description='The x variable',
            size=5,
            bounds=None,
            domain=None,
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x8',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': 'error',
                'bounds': None,
                'domain': None,
            }
        )
        x9 = f.Variable(
            name='x9',
            guess=1.0,
            units='m',
            description='The x variable',
            size=[2, 2],
            bounds=None,
            domain=None,
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x10',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': ['2', '2'],
                'bounds': None,
                'domain': None,
            }
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x11',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': [2, 1],
                'bounds': None,
                'domain': None,
            }
        )

        x12 = f.Variable(
            name='x12',
            guess=1.0,
            units='m',
            description='The x variable',
            size=None,
            bounds=[-10, 10],
            domain=None,
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x13',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'bounds': [10, -10],
                'domain': None,
            }
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x14',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'bounds': ["-10", "10"],
                'domain': None,
            }
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x15',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'bounds': [1, 2, 3],
                'domain': None,
            }
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x16',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'bounds': "error",
                'domain': None,
            }
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x17',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'bounds': [0, "10"],
                'domain': None,
            }
        )

        # verifies alternate unit construction
        x18 = f.Variable('x18', 1.0, pyo.units.m)
        self.assertRaises(AttributeError, f.Variable, *('x19', 1.0, 'string'))

    def test_edi_formulation_constant(self):
        "Tests the constant constructor in edi.formulation"
        from pyomo.contrib.edi import Formulation
        from pyomo.environ import Reals, PositiveReals

        f = Formulation()

        c1 = f.Constant(
            name='c1',
            value=1.0,
            units='m',
            description='A constant c',
            size=None,
            within=None,
        )
        self.assertRaises(RuntimeError, f.Constant, *('c1', 1.0, 'm'))
        c2 = f.Constant('c2', 1.0, 'm')
        c3 = f.Constant('c3', 1.0, 'm', 'A constant c', None, None)
        c4 = f.Constant(
            name='c4',
            value=1.0,
            units='m',
            description='A constant c',
            size=None,
            within=PositiveReals,
        )
        self.assertRaises(
            RuntimeError,
            f.Constant,
            **{
                'name': 'c5',
                'value': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'within': "error",
            }
        )

        c6 = f.Constant(
            name='c6',
            value=1.0,
            units='m',
            description='A constant c',
            size=0,
            within=None,
        )
        c7 = f.Constant(
            name='c7',
            value=1.0,
            units='m',
            description='A constant c',
            size=5,
            within=None,
        )
        self.assertRaises(
            ValueError,
            f.Constant,
            **{
                'name': 'c8',
                'value': 1.0,
                'units': 'm',
                'description': 'A constant c',
                'size': 'error',
                'within': None,
            }
        )
        c9 = f.Constant(
            name='c9',
            value=1.0,
            units='m',
            description='A constant c',
            size=[2, 2],
            within=None,
        )
        self.assertRaises(
            ValueError,
            f.Constant,
            **{
                'name': 'c10',
                'value': 1.0,
                'units': 'm',
                'description': 'A constant c',
                'size': ['2', '2'],
                'within': None,
            }
        )
        self.assertRaises(
            ValueError,
            f.Constant,
            **{
                'name': 'c11',
                'value': 1.0,
                'units': 'm',
                'description': 'A constant c',
                'size': [2, 1],
                'within': None,
            }
        )

    def test_edi_formulation_objective(self):
        "Tests the objective constructor in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        f.Objective(x + y)

    def test_edi_formulation_runtimeobjective(self):
        "Tests the runtime objective constructor in edi.formulation"
        # TODO: not currently implemented, see:  https://github.com/codykarcher/pyomo/issues/5
        pass

    def test_edi_formulation_constraint(self):
        "Tests the constraint constructor in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        f.Objective(x + y)
        f.Constraint(x + y <= 1.0 * units.m)

    def test_edi_formulation_runtimeconstraint_tuple(self):
        "Tests the runtime constraint constructor in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(
            name='z', guess=1.0, units='m^2', description='The unit circle output'
        )
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=2
        )
        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super(UnitCircle, self).__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')
                self.outputs.append(
                    name='z',
                    units='ft**2',
                    description='Resultant of the unit circle evaluation',
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = pyo.value(
                    units.convert(y, self.inputs[1].units)
                )  # Converts to correct units then casts to float
                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.Constraint(z <= 1 * units.m**2)

        f.RuntimeConstraint(*(z, '==', [x, y], UnitCircle()))

    def test_edi_formulation_runtimeconstraint_list(self):
        "Tests the runtime constraint constructor in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(
            name='z', guess=1.0, units='m^2', description='The unit circle output'
        )
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=2
        )
        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super(UnitCircle, self).__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')
                self.outputs.append(
                    name='z',
                    units='ft**2',
                    description='Resultant of the unit circle evaluation',
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = pyo.value(
                    units.convert(y, self.inputs[1].units)
                )  # Converts to correct units then casts to float
                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.Constraint(z <= 1 * units.m**2)

        f.RuntimeConstraint(*[[z], ['=='], [x, y], UnitCircle()])

    def test_edi_formulation_runtimeconstraint_dict(self):
        "Tests the runtime constraint constructor in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(
            name='z', guess=1.0, units='m^2', description='The unit circle output'
        )
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=2
        )
        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super(UnitCircle, self).__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')
                self.outputs.append(
                    name='z',
                    units='ft**2',
                    description='Resultant of the unit circle evaluation',
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = pyo.value(
                    units.convert(y, self.inputs[1].units)
                )  # Converts to correct units then casts to float
                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.Constraint(z <= 1 * units.m**2)

        f.RuntimeConstraint(
            **{
                'outputs': z,
                'operators': '==',
                'inputs': [x, y],
                'black_box': UnitCircle(),
            }
        )

    def test_edi_formulation_constraintlist_1(self):
        "Tests the constraint list constructor in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(
            name='z', guess=1.0, units='m^2', description='The unit circle output'
        )
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=2
        )
        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super(UnitCircle, self).__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')
                self.outputs.append(
                    name='z',
                    units='ft**2',
                    description='Resultant of the unit circle evaluation',
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = pyo.value(
                    units.convert(y, self.inputs[1].units)
                )  # Converts to correct units then casts to float
                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.ConstraintList([(z, '==', [x, y], UnitCircle()), z <= 1 * units.m**2])

        cl = f.get_constraints()

        self.assertTrue(len(cl) == 2)

    def test_edi_formulation_constraintlist_2(self):
        "Tests the constraint list constructor in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m**2', description='The y variable')
        f.Objective(y)

        class Parabola(BlackBoxFunctionModel):
            def __init__(self):
                super(Parabola, self).__init__()
                self.description = 'This model evaluates the function: y = x**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.outputs.append(
                    name='y', units='ft**2', description='The y variable'
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = x**2  # Compute y
                dydx = 2 * x  # Compute dy/dx
                y *= units.ft**2
                dydx *= units.ft  # units.ft**2 / units.ft
                return y, [dydx]  # return z, grad(z), hess(z)...

        f.ConstraintList(
            [{'outputs': y, 'operators': '==', 'inputs': x, 'black_box': Parabola()}]
        )

        cl = f.get_constraints()

        self.assertTrue(len(cl) == 1)

    def test_edi_formulation_constraintlist_3(self):
        "Tests the constraint list constructor in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(
            name='x', guess=1.0, units='m', description='The x variable', size=3
        )
        y = f.Variable(
            name='y', guess=1.0, units='m**2', description='The y variable', size=3
        )
        f.Objective(y[0] + y[1] + y[2])

        class Parabola(BlackBoxFunctionModel):
            def __init__(self):
                super(Parabola, self).__init__()
                self.description = 'This model evaluates the function: y = x**2'
                self.inputs.append(
                    name='x', size=3, units='ft', description='The x variable'
                )
                self.outputs.append(
                    name='y', size=3, units='ft**2', description='The y variable'
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(*args, **kwargs):  # The actual function that does things
                args = list(args)
                self = args.pop(0)
                runCases, returnMode, remainingKwargs = self.parseInputs(
                    *args, **kwargs
                )

                x = self.sanitizeInputs(runCases[0]['x'])
                x = np.array([pyo.value(xval) for xval in x], dtype=np.float64)

                y = x**2  # Compute y
                dydx = 2 * x  # Compute dy/dx

                y = y * units.ft**2
                dydx = np.diag(dydx)
                dydx = dydx * units.ft  # units.ft**2 / units.ft

                return y, [dydx]  # return z, grad(z), hess(z)...

        f.ConstraintList(
            [{'outputs': y, 'operators': '==', 'inputs': x, 'black_box': Parabola()}]
        )
        cl = f.get_constraints()
        self.assertTrue(len(cl) == 1)

    def test_edi_formulation_runtimeconstraint_exceptions(self):
        "Tests the runtime constraint constructor in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(
            name='z', guess=1.0, units='m^2', description='The unit circle output'
        )
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=2
        )
        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super(UnitCircle, self).__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')
                self.outputs.append(
                    name='z',
                    units='ft**2',
                    description='Resultant of the unit circle evaluation',
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = pyo.value(
                    units.convert(y, self.inputs[1].units)
                )  # Converts to correct units then casts to float
                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.Constraint(z <= 1 * units.m**2)

        # flaggs the input or  as bad before assigning the incorrect black box
        self.assertRaises(
            ValueError, f.RuntimeConstraint, *(z, '==', 1.0, UnitCircle())
        )
        self.assertRaises(
            ValueError, f.RuntimeConstraint, *(1.0, '==', [x, y], UnitCircle())
        )

        self.assertRaises(
            ValueError, f.RuntimeConstraint, *(z, '==', [1.0, y], UnitCircle())
        )
        self.assertRaises(
            ValueError, f.RuntimeConstraint, *(z, 1.0, [x, y], UnitCircle())
        )
        self.assertRaises(
            ValueError, f.RuntimeConstraint, *(z, '=', [x, y], UnitCircle())
        )

    def test_edi_formulation_getvariables(self):
        "Tests the get_variables function in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')

        vrs = f.get_variables()
        self.assertListEqual(vrs, [x, y])

    def test_edi_formulation_getconstants(self):
        "Tests the get_constants function in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')

        c1 = f.Constant(
            name='c1',
            value=1.0,
            units='m',
            description='A constant c1',
            size=None,
            within=None,
        )
        c2 = f.Constant(
            name='c2',
            value=1.0,
            units='m',
            description='A constant c2',
            size=None,
            within=None,
        )

        csts = f.get_constants()
        self.assertListEqual(csts, [c1, c2])

    def test_edi_formulation_getobjectives(self):
        "Tests the get_objectives function in edi.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        f.Objective(x + y)
        objList = f.get_objectives()
        self.assertTrue(len(objList) == 1)
        # not really sure how to check this, so I wont

    def test_edi_formulation_getconstraints(self):
        "Tests the get_constraints, get_explicitConstraints, and get_runtimeConstraints functions in edi.formulation"
        # =================
        # Import Statements
        # =================
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        # ===================
        # Declare Formulation
        # ===================
        f = Formulation()

        # =================
        # Declare Variables
        # =================
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(
            name='z', guess=1.0, units='m^2', description='The unit circle output'
        )

        # =================
        # Declare Constants
        # =================
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=2
        )

        # =====================
        # Declare the Objective
        # =====================
        f.Objective(c[0] * x + c[1] * y)

        # ===================
        # Declare a Black Box
        # ===================
        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):  # The initialization function
                # Initialize the black box model
                super(UnitCircle, self).__init__()

                # A brief description of the model
                self.description = 'This model evaluates the function: z = x**2 + y**2'

                # Declare the black box model inputs
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')

                # Declare the black box model outputs
                self.outputs.append(
                    name='z',
                    units='ft**2',
                    description='Resultant of the unit circle evaluation',
                )

                # Declare the maximum available derivative
                self.availableDerivative = 1

                # Post-initialization setup
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = pyo.value(
                    units.convert(y, self.inputs[1].units)
                )  # Converts to correct units then casts to float

                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy

                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft

                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        # =======================
        # Declare the Constraints
        # =======================
        f.ConstraintList([(z, '==', [x, y], UnitCircle()), z <= 1 * units.m**2])

        cl = f.get_constraints()
        ecl = f.get_explicitConstraints()
        rcl = f.get_runtimeConstraints()

        self.assertTrue(len(cl) == 2)
        self.assertTrue(len(ecl) == 1)
        self.assertTrue(len(rcl) == 1)

    def test_edi_formulation_checkunits(self):
        "Tests the check_units function in edi.formulation"
        import pyomo
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')

        f.Objective(x + y)
        f.Constraint(x + y <= 1.0 * units.m)
        f.check_units()

        f.Constraint(2.0 * x + y <= 1.0)
        self.assertRaises(
            pyomo.core.base.units_container.UnitsError, f.check_units, *()
        )

        f2 = Formulation()
        u = f2.Variable(name='u', guess=1.0, units='m', description='The u variable')
        v = f2.Variable(name='v', guess=1.0, units='kg', description='The v variable')
        f2.Objective(u + v)
        self.assertRaises(
            pyomo.core.base.units_container.UnitsError, f2.check_units, *()
        )


if __name__ == '__main__':
    unittest.main()
