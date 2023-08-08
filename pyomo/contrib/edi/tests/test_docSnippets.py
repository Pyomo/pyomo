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

from pyomo.core.base.units_container import (
    pint_available,
)

from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.dependencies import scipy, scipy_available
egb, egb_available = attempt_import("pyomo.contrib.pynumero.interfaces.external_grey_box")

if numpy_available:
    import numpy as np

@unittest.skipIf(not egb_available,   'Testing pyomo.contrib.edi requires pynumero external grey boxes')
@unittest.skipIf(not numpy_available, 'Testing pyomo.contrib.edi requires numpy')
@unittest.skipIf(not scipy_available, 'Testing pyomo.contrib.edi requires scipy')
@unittest.skipIf(not pint_available,  'Testing units requires pint')
class TestEDISnippets(unittest.TestCase):
    def test_edi_snippet_formuation_01(self):
        # BEGIN: Formulation_Snippet_01
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        # END: Formulation_Snippet_01

    def test_edi_snippet_formuation_02(self):
        # BEGIN: Formulation_Snippet_02
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='')
        # END: Formulation_Snippet_02

    def test_edi_snippet_formuation_03(self):
        # BEGIN: Formulation_Snippet_03
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        c = f.Constant(name='c', value=1.0, units='')
        # END: Formulation_Snippet_03

    def test_edi_snippet_formuation_04(self):
        # BEGIN: Formulation_Snippet_04
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='')
        y = f.Variable(name='y', guess=1.0, units='')
        c = f.Constant(name='c', value=1.0, units='')
        f.Objective(c * x + y)
        # END: Formulation_Snippet_04

    def test_edi_snippet_formuation_05(self):
        # BEGIN: Formulation_Snippet_05
        from pyomo.contrib.edi import Formulation
        from pyomo.environ import maximize, minimize

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='')
        y = f.Variable(name='y', guess=1.0, units='')
        c = f.Constant(name='c', value=1.0, units='')
        f.Objective(c * x + y, sense=maximize)
        # END: Formulation_Snippet_05

    def test_edi_snippet_formuation_06(self):
        # BEGIN: Formulation_Snippet_06
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='')
        y = f.Variable(name='y', guess=1.0, units='')
        c = f.Constant(name='c', value=1.0, units='')
        f.Objective(c * x + y)
        f.Constraint(x**2 + y**2 <= 1.0)
        f.Constraint(x >= 0)
        f.Constraint(y <= 0)
        # END: Formulation_Snippet_06

    def test_edi_snippet_formuation_07(self):
        # BEGIN: Formulation_Snippet_07
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='')
        y = f.Variable(name='y', guess=1.0, units='')
        c = f.Constant(name='c', value=1.0, units='')
        f.Objective(c * x + y)
        f.ConstraintList([x**2 + y**2 <= 1.0, x >= 0, y <= 0])
        # END: Formulation_Snippet_07

    def test_edi_snippet_formuation_08(self):
        # BEGIN: Formulation_Snippet_08
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='')
        y = f.Variable(name='y', guess=1.0, units='')
        c = f.Constant(name='c', value=1.0, units='')
        f.Objective(c * x + y)

        constraintList = [x**2 + y**2 <= 1.0, x >= 0, y <= 0]

        f.ConstraintList(constraintList)
        # END: Formulation_Snippet_08

    def test_edi_snippet_formuation_09(self):
        # BEGIN: Formulation_Snippet_09
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='y variable')
        z = f.Variable(name='z', guess=1.0, units='m^2', description='Output var')
        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super().__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')
                self.outputs.append(
                    name='z', units='ft**2', description='Output variable'
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                # Converts to correct units then casts to float
                x = pyo.value(units.convert(x, self.inputs['x'].units))
                y = pyo.value(units.convert(y, self.inputs['y'].units))
                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.Constraint(z <= 1 * units.m**2)

        f.RuntimeConstraint(z, '==', [x, y], UnitCircle())
        # END: Formulation_Snippet_09

        ### This will fail validatation, but should construct appropriately

        # BEGIN: Formulation_Snippet_10
        f.RuntimeConstraint(*(z, '==', [x, y], UnitCircle()))
        # END: Formulation_Snippet_10

        # BEGIN: Formulation_Snippet_11
        f.RuntimeConstraint(*[z, '==', [x, y], UnitCircle()])
        # END: Formulation_Snippet_11

        # BEGIN: Formulation_Snippet_12
        f.RuntimeConstraint(
            **{
                'outputs': z,
                'operators': '==',
                'inputs': [x, y],
                'black_box': UnitCircle(),
            }
        )
        # END: Formulation_Snippet_12

        # BEGIN: Formulation_Snippet_13
        f.RuntimeConstraint(*([z], ['=='], [x, y], UnitCircle()))
        # END: Formulation_Snippet_13

    def test_edi_snippet_formuation_14(self):
        # BEGIN: Formulation_Snippet_14
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='y variable')
        z = f.Variable(name='z', guess=1.0, units='m^2', description='Output var')
        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super().__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')
                self.outputs.append(
                    name='z', units='ft**2', description='Output variable'
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                # Converts to correct units then casts to float
                x = pyo.value(units.convert(x, self.inputs[0].units))
                y = pyo.value(units.convert(y, self.inputs[1].units))
                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.ConstraintList([z <= 1 * units.m**2, [z, '==', [x, y], UnitCircle()]])
        # END: Formulation_Snippet_14

        ### This will fail validatation, but should construct appropriately

        # BEGIN: Formulation_Snippet_15
        f.ConstraintList([z <= 1 * units.m**2, (z, '==', [x, y], UnitCircle())])
        # END: Formulation_Snippet_15

        # BEGIN: Formulation_Snippet_16
        f.ConstraintList([z <= 1 * units.m**2, [z, '==', [x, y], UnitCircle()]])
        # END: Formulation_Snippet_16

        # BEGIN: Formulation_Snippet_17
        f.ConstraintList(
            [
                z <= 1 * units.m**2,
                {
                    'outputs': z,
                    'operators': '==',
                    'inputs': [x, y],
                    'black_box': UnitCircle(),
                },
            ]
        )
        # END: Formulation_Snippet_17

        # BEGIN: Formulation_Snippet_18
        f.ConstraintList([z <= 1 * units.m**2, ([z], ['=='], [x, y], UnitCircle())])
        # END: Formulation_Snippet_18

    def test_edi_snippet_variables_01(self):
        # BEGIN: Variables_Snippet_01
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        # END: Variables_Snippet_01

    def test_edi_snippet_variables_02(self):
        # BEGIN: Variables_Snippet_02
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable('x', 1.0, 'm')
        # END: Variables_Snippet_02

    def test_edi_snippet_variables_03(self):
        # BEGIN: Variables_Snippet_03
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(
            name='x',
            guess=1.0,
            units='m',
            description='The x variable',
            bounds=[-10, 10],
        )
        # END: Variables_Snippet_03

    def test_edi_snippet_variables_04(self):
        # BEGIN: Variables_Snippet_04
        from pyomo.contrib.edi import Formulation
        from pyomo.environ import Integers

        f = Formulation()
        x = f.Variable(
            name='x',
            guess=1.0,
            units='m',
            description='The x variable',
            domain=Integers,
        )
        # END: Variables_Snippet_04

    def test_edi_snippet_variables_05(self):
        # BEGIN: Variables_Snippet_05
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units=units.m, description='The x variable')
        # END: Variables_Snippet_05

    def test_edi_snippet_variables_06(self):
        # BEGIN: Variables_Snippet_06
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(
            name='x', guess=1.0, units='m', description='The x variable', size=5
        )
        # END: Variables_Snippet_06

    def test_edi_snippet_variables_07(self):
        # BEGIN: Variables_Snippet_07
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(
            name='x', guess=1.0, units='m', description='The x variable', size=[10, 2]
        )
        # END: Variables_Snippet_07

    def test_edi_snippet_variables_08(self):
        # BEGIN: Variables_Snippet_08
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(
            name='x', guess=1.0, units='kg*m/s**2', description='The x variable'
        )
        # END: Variables_Snippet_08

    def test_edi_snippet_constants_01(self):
        # BEGIN: Constants_Snippet_01
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Constant(name='c', value=1.0, units='m', description='A constant c')
        # END: Constants_Snippet_01

    def test_edi_snippet_constants_02(self):
        # BEGIN: Constants_Snippet_02
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Constant('c', 1.0, 'm')
        # END: Constants_Snippet_02

    def test_edi_snippet_constants_03(self):
        # BEGIN: Constants_Snippet_03
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Constant(name='c', value=1.0, units=units.m, description='A constant c')
        # END: Constants_Snippet_03

    def test_edi_snippet_constants_04(self):
        # BEGIN: Constants_Snippet_04
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Constant(
            name='c', value=1.0, units='m', description='A constant c', size=5
        )
        # END: Constants_Snippet_04

    def test_edi_snippet_constants_05(self):
        # BEGIN: Constants_Snippet_05
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Constant(
            name='c', value=1.0, units='m', description='A constant c', size=[10, 2]
        )
        # END: Constants_Snippet_05

    def test_edi_snippet_constants_06(self):
        # BEGIN: Constants_Snippet_06
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Constant(
            name='c', value=1.0, units='kg*m/s**2', description='A constant c'
        )
        # END: Constants_Snippet_06

    def test_edi_snippet_objectives_01(self):
        # BEGIN: Objectives_Snippet_01
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        c = f.Constant(name='c', value=1.0, units='', description='A constant c')
        f.Objective(c * x + y)  # Default is minimize
        # END: Objectives_Snippet_01

    def test_edi_snippet_objectives_02(self):
        # BEGIN: Objectives_Snippet_02
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        c = f.Constant(name='c', value=1.0, units='', description='A constant c')
        f.Objective(c * x**4 + y**4)  # Default is minimize
        # END: Objectives_Snippet_02

    def test_edi_snippet_objectives_03(self):
        # BEGIN: Objectives_Snippet_03
        from pyomo.contrib.edi import Formulation
        from pyomo.environ import minimize, maximize

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        c = f.Constant(name='c', value=1.0, units='', description='A constant c')
        f.Objective(c * x**4 + y**4, sense=minimize)
        # END: Objectives_Snippet_03

    def test_edi_snippet_objectives_04(self):
        # BEGIN: Objectives_Snippet_04
        from pyomo.contrib.edi import Formulation
        from pyomo.environ import minimize, maximize

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        c = f.Constant(name='c', value=1.0, units='', description='A constant c')
        f.Objective(c * x**4 + y**4, sense=1)  # 1 corresponds to minimize
        # END: Objectives_Snippet_04

    def test_edi_snippet_objectives_05(self):
        # BEGIN: Objectives_Snippet_05
        from pyomo.contrib.edi import Formulation
        from pyomo.environ import minimize, maximize

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        c = f.Constant(name='c', value=1.0, units='', description='A constant c')
        f.Objective(-c * x**4 - y**4, sense=maximize)
        # END: Objectives_Snippet_05

    def test_edi_snippet_objectives_06(self):
        # BEGIN: Objectives_Snippet_06
        from pyomo.contrib.edi import Formulation
        from pyomo.environ import minimize, maximize

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        c = f.Constant(name='c', value=1.0, units='', description='A constant c')
        f.Objective(-c * x**4 - y**4, sense=-1)  # -1 corresponds to maximize
        # END: Objectives_Snippet_06

    def test_edi_snippet_objectives_07(self):
        # BEGIN: Objectives_Snippet_07
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(
            name='x',
            guess=1.0,
            units='m',
            description='The x variable',
            bounds=[0, 100],
            size=3,
        )
        y = f.Variable(
            name='y', guess=1.0, units='m', description='The y variable', size=[2, 2]
        )
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=3
        )
        f.Objective(
            c[0] * x[0]
            + c[1] * x[1]
            + c[2] * x[2]
            + y[0, 0] ** 4
            + y[0, 1] ** 4
            + y[1, 0] ** 4
            + y[1, 1] ** 4
        )  # Default is minimize
        # END: Objectives_Snippet_07

    def test_edi_snippet_constraints_01(self):
        # BEGIN: Constraints_Snippet_01
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        c = f.Constant(name='c', value=1.0, units='', description='A constant c')
        f.Objective(c * x + y)
        f.ConstraintList(
            [x**2 + y**2 <= 1.0 * units.m**2, x <= 0.75 * units.m, x >= y]
        )
        # END: Constraints_Snippet_01

    def test_edi_snippet_constraints_02(self):
        # BEGIN: Constraints_Snippet_02
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        c = f.Constant(name='c', value=1.0, units='', description='A constant c')
        f.Objective(c * x + y)
        f.Constraint(x**2 + y**2 <= 1.0 * units.m**2)
        f.Constraint(x <= 0.75 * units.m)
        f.Constraint(x >= y)
        # END: Constraints_Snippet_02

    def test_edi_snippet_constraints_03(self):
        # BEGIN: Constraints_Snippet_03
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation

        f = Formulation()
        x = f.Variable(
            name='x',
            guess=1.0,
            units='m',
            description='The x variable',
            bounds=[0, 100],
            size=3,
        )
        y = f.Variable(
            name='y', guess=1.0, units='m', description='The y variable', size=[2, 2]
        )
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=3
        )
        f.Objective(
            c[0] * x[0]
            + c[1] * x[1]
            + c[2] * x[2]
            + y[0, 0] ** 4
            + y[0, 1] ** 4
            + y[1, 0] ** 4
            + y[1, 1] ** 4
        )  # Default is minimize
        f.ConstraintList(
            [
                x[0] ** 2 + x[1] ** 2 + x[2] ** 2 <= 1.0 * units.m,
                y[0, 0] >= 1.0 * units.m,
                y[0, 1] >= 1.0 * units.m,
                y[1, 0] >= 1.0 * units.m,
                y[1, 1] >= 1.0 * units.m,
                x[0] >= y[0, 0],
            ]
        )
        # END: Constraints_Snippet_03

    def test_edi_snippet_runtimeconstraints_01(self):
        # BEGIN: RuntimeConstraints_Snippet_01
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import BlackBoxFunctionModel

        class Parabola(BlackBoxFunctionModel):
            def __init__(self):
                # Call parent init
                super().__init__()

                # A brief description of the model
                self.description = 'This model evaluates the function: y = x**2'

                # Append the model inputs
                self.inputs.append(name='x', units='ft', description='The x variable')

                # Append the model outputs
                self.outputs.append(
                    name='y', units='ft**2', description='The y variable'
                )

                # Set the highest available derivative
                # Should be 1 for most cases but defaults to 0
                self.availableDerivative = 1

            def BlackBox(self, x):  # The actual function that does things
                # Convert to correct units and cast to a float
                x = pyo.value(units.convert(x, self.inputs['x'].units))

                # Compute y
                y = x**2

                # Compute dy/dx
                dydx = 2 * x

                # Add the units to the output
                y = y * self.outputs['y'].units

                # Add the units to the derivative for output
                dydx = dydx * self.outputs['y'].units / self.inputs['x'].units

                # Return using the output packing guidelines described in the documentation:
                #     returnVal[0]    = output_value
                #     returnVal[1]    = jacobian
                #     returnVal[1][0] = derivative_scalarOutput_wrt_0th_input
                return y, [dydx]

        # END: RuntimeConstraints_Snippet_01

    def test_edi_snippet_runtimeconstraints_02(self):
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import BlackBoxFunctionModel

        class Parabola(BlackBoxFunctionModel):
            def __init__(self):
                # BEGIN: RuntimeConstraints_Snippet_02
                # Call parent init
                super().__init__()
                # END: RuntimeConstraints_Snippet_02

                # A brief description of the model
                self.description = 'This model evaluates the function: y = x**2'

                # Append the model inputs
                self.inputs.append(name='x', units='ft', description='The x variable')

                # Append the model outputs
                self.outputs.append(
                    name='y', units='ft**2', description='The y variable'
                )

                # BEGIN: RuntimeConstraints_Snippet_05
                # Set the highest available derivative
                # Should be 1 for most cases but defaults to 0
                self.availableDerivative = 1
                # END: RuntimeConstraints_Snippet_05

            def BlackBox(self, x):  # The actual function that does things
                storeX = x

                # BEGIN: RuntimeConstraints_Snippet_06
                x = units.convert(x, self.inputs['x'].units)
                # END: RuntimeConstraints_Snippet_06

                x = storeX

                # BEGIN: RuntimeConstraints_Snippet_07
                # Convert to correct units and cast to a float
                x = pyo.value(units.convert(x, self.inputs['x'].units))
                # END: RuntimeConstraints_Snippet_07

                # Compute y
                y = x**2

                # Compute dy/dx
                dydx = 2 * x

                # BEGIN: RuntimeConstraints_Snippet_08
                # Add the units to the output
                y = y * self.outputs['y'].units

                # Add the units to the derivative for output
                dydx = dydx * self.outputs['y'].units / self.inputs['x'].units
                # END: RuntimeConstraints_Snippet_08

                # BEGIN: RuntimeConstraints_Snippet_09
                # Return using the output packing guidelines described in the documentation:
                #     returnVal[0]    = output_value
                #     returnVal[1]    = jacobian
                #     returnVal[1][0] = derivative_scalarOutput_wrt_0th_input
                return y, [dydx]
                # END: RuntimeConstraints_Snippet_09

    def test_edi_snippet_runtimeconstraints_03(self):
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(name='z', guess=1.0, units='m^2', description='The z variable')

        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super().__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                # BEGIN: RuntimeConstraints_Snippet_03
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')
                # END: RuntimeConstraints_Snippet_03
                self.outputs.append(
                    name='z',
                    units='ft**2',
                    description='Resultant of the unit circle evaluation',
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                x = pyo.value(units.convert(x, self.inputs['x'].units))
                y = pyo.value(units.convert(y, self.inputs['y'].units))
                z = x**2 + y**2
                dzdx = 2 * x
                dzdy = 2 * y
                z = z * self.outputs['z'].units
                dzdx = dzdx * self.outputs['z'].units / self.inputs['x'].units
                dzdy = dzdy * self.outputs['z'].units / self.inputs['y'].units
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.ConstraintList([(z, '==', [x, y], UnitCircle()), z <= 1 * units.m**2])

    def test_edi_snippet_runtimeconstraints_04(self):
        import numpy as np
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel
        from pyomo.common.formatting import tostr

        class PassThrough(BlackBoxFunctionModel):
            def __init__(self):
                # Set up all the attributes by calling Model.__init__
                super().__init__()

                # Setup Inputs
                self.inputs.append('x', '', 'X Variable')
                self.inputs.append('y', '', 'Y Variable')

                # BEGIN: RuntimeConstraints_Snippet_04
                # Setup Outputs
                self.outputs.append('u', '', 'U Variable')
                self.outputs.append('v', '', 'V Variable')
                # END: RuntimeConstraints_Snippet_04

                # Simple model description
                self.description = 'This model is a pass through)'

                self.availableDerivative = 1

            # standard function call is y(, dydx, ...) = self.BlackBox(**{'x1':x1, 'x2':x2, ...})
            def BlackBox(self, *args, **kwargs):
                runCases, returnMode, extras = self.parseInputs(*args, **kwargs)
                x = [
                    self.pyomo_value(runCases[i]['x']) for i in range(0, len(runCases))
                ]
                y = [
                    self.pyomo_value(runCases[i]['x']) for i in range(0, len(runCases))
                ]

                u = []
                dudx = []
                dudy = []
                v = []
                dvdx = []
                dvdy = []

                for xval in x:
                    u.append(xval * units.dimensionless)
                    dudx.append(1.0 * units.dimensionless)
                    dudy.append(0.0 * units.dimensionless)

                for yval in y:
                    v.append(yval * units.dimensionless)
                    dvdx.append(0.0 * units.dimensionless)
                    dvdy.append(1.0 * units.dimensionless)

                if returnMode < 0:
                    returnMode = -1 * (returnMode + 1)
                    if returnMode == 0:
                        return [u[0], v[0]]
                    if returnMode == 1:
                        return [u[0], v[0]], [[dudx[0], dudy[0]], [dvdx[0], dvdy[0]]]
                else:
                    if returnMode == 0:
                        opt = []
                        for i in range(0, len(y)):
                            opt.append([u[i], v[i]])
                        return opt
                    if returnMode == 1:
                        opt = []
                        for i in range(0, len(y)):
                            opt.append(
                                [[u[i], v[i]], [[dudx[i], dudy[i]], [dvdx[i], dvdy[i]]]]
                            )
                        return opt

        bb = PassThrough()
        bbo = bb.BlackBox(1.0, 1.0)
        bbo = bb.BlackBox({'x': np.linspace(0, 10, 11), 'y': np.linspace(0, 10, 11)})

    def test_edi_snippet_runtimeconstraints_10(self):
        # BEGIN: RuntimeConstraints_Snippet_10
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(name='z', guess=1.0, units='m^2', description='The z variable')

        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super().__init__()
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
                x = pyo.value(units.convert(x, self.inputs['x'].units))
                y = pyo.value(units.convert(y, self.inputs['y'].units))
                z = x**2 + y**2
                dzdx = 2 * x
                dzdy = 2 * y
                z = z * self.outputs['z'].units
                dzdx = dzdx * self.outputs['z'].units / self.inputs['x'].units
                dzdy = dzdy * self.outputs['z'].units / self.inputs['y'].units
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.ConstraintList([(z, '==', [x, y], UnitCircle()), z <= 1 * units.m**2])
        # END: RuntimeConstraints_Snippet_10

    def test_edi_snippet_advancedRTC_01(self):
        # BEGIN: AdvancedRTC_Snippet_01
        import numpy as np
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        class SignomialTest(BlackBoxFunctionModel):
            def __init__(self):
                # Set up all the attributes by calling Model.__init__
                super().__init__()

                # Setup Inputs
                self.inputs.append('x', '', 'Independent Variable')

                # Setup Outputs
                self.outputs.append('y', '', 'Dependent Variable')

                # Simple model description
                self.description = (
                    'This model evaluates the ' + 'function: max([-6*x-6, x**4-3*x**2])'
                )

                self.availableDerivative = 1

            def BlackBox(self, *args, **kwargs):
                runCases, returnMode, extras = self.parseInputs(*args, **kwargs)

                x = np.array(
                    [pyo.value(runCases[i]['x']) for i in range(0, len(runCases))]
                )

                y = np.maximum(-6 * x - 6, x**4 - 3 * x**2)
                dydx = 4 * x**3 - 6 * x
                ddy_ddx = 12 * x**2 - 6
                gradientSwitch = -6 * x - 6 > x**4 - 3 * x**2
                dydx[gradientSwitch] = -6
                ddy_ddx[gradientSwitch] = 0

                y = [yval * units.dimensionless for yval in y]
                dydx = [dydx[i] * units.dimensionless for i in range(0, len(dydx))]

                if returnMode < 0:
                    returnMode = -1 * (returnMode + 1)
                    if returnMode == 0:
                        return y[0]
                    if returnMode == 1:
                        return y[0], dydx
                else:
                    if returnMode == 0:
                        opt = []
                        for i in range(0, len(y)):
                            opt.append([y[i]])
                        return opt
                    if returnMode == 1:
                        opt = []
                        for i in range(0, len(y)):
                            opt.append([[y[i]], [[[dydx[i]]]]])
                        return opt

        s = SignomialTest()
        ivals = [[x] for x in np.linspace(-2, 2, 11)]

        # How the black box may be called using EDI
        bbo = s.BlackBox(**{'x': 0.5})
        bbo = s.BlackBox({'x': 0.5})
        bbo = s.BlackBox(**{'x': 0.5, 'optn': True})

        # Additional options available with parseInputs
        bbo = s.BlackBox(*[0.5], **{'optn1': True, 'optn2': False})
        bbo = s.BlackBox(*[0.5, True], **{'optn': False})
        bbo = s.BlackBox({'x': [x for x in np.linspace(-2, 2, 11)]})
        bbo = s.BlackBox([{'x': x} for x in np.linspace(-2, 2, 11)])
        bbo = s.BlackBox([[x] for x in np.linspace(-2, 2, 11)])
        bbo = s.BlackBox([[x] for x in np.linspace(-2, 2, 11)], True, optn=False)
        bbo = s.BlackBox([[x] for x in np.linspace(-2, 2, 11)], optn1=True, optn2=False)
        # END: AdvancedRTC_Snippet_01

    def test_edi_snippet_advancedRTC_02(self):
        import numpy as np
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        class SignomialTest(BlackBoxFunctionModel):
            def __init__(self):
                # Set up all the attributes by calling Model.__init__
                super().__init__()

                # Setup Inputs
                self.inputs.append('x', '', 'Independent Variable')

                # Setup Outputs
                self.outputs.append('y', '', 'Dependent Variable')

                # Simple model description
                self.description = (
                    'This model evaluates the ' + 'function: max([-6*x-6, x**4-3*x**2])'
                )

                self.availableDerivative = 1

            # BEGIN: AdvancedRTC_Snippet_02
            def BlackBox(self, *args, **kwargs):
                runCases, returnMode, extras = self.parseInputs(*args, **kwargs)
                # END: AdvancedRTC_Snippet_02

                x = np.array(
                    [pyo.value(runCases[i]['x']) for i in range(0, len(runCases))]
                )

                y = np.maximum(-6 * x - 6, x**4 - 3 * x**2)
                dydx = 4 * x**3 - 6 * x
                ddy_ddx = 12 * x**2 - 6
                gradientSwitch = -6 * x - 6 > x**4 - 3 * x**2
                dydx[gradientSwitch] = -6
                ddy_ddx[gradientSwitch] = 0

                y = [yval * units.dimensionless for yval in y]
                dydx = [dydx[i] * units.dimensionless for i in range(0, len(dydx))]

                if returnMode < 0:
                    returnMode = -1 * (returnMode + 1)
                    if returnMode == 0:
                        return y[0]
                    if returnMode == 1:
                        return y[0], dydx
                else:
                    if returnMode == 0:
                        opt = []
                        for i in range(0, len(y)):
                            opt.append([y[i]])
                        return opt
                    if returnMode == 1:
                        opt = []
                        for i in range(0, len(y)):
                            opt.append([[y[i]], [[[dydx[i]]]]])
                        return opt

        s = SignomialTest()
        ivals = [[x] for x in np.linspace(-2, 2, 11)]

        # How the black box may be called using EDI
        bbo = s.BlackBox(**{'x': 0.5})
        bbo = s.BlackBox({'x': 0.5})
        bbo = s.BlackBox(**{'x': 0.5, 'optn': True})

        # Additional options available with parseInputs
        bbo = s.BlackBox(*[0.5], **{'optn1': True, 'optn2': False})
        bbo = s.BlackBox(*[0.5, True], **{'optn': False})
        bbo = s.BlackBox({'x': [x for x in np.linspace(-2, 2, 11)]})
        bbo = s.BlackBox([{'x': x} for x in np.linspace(-2, 2, 11)])
        bbo = s.BlackBox([[x] for x in np.linspace(-2, 2, 11)])
        bbo = s.BlackBox([[x] for x in np.linspace(-2, 2, 11)], True, optn=False)
        bbo = s.BlackBox([[x] for x in np.linspace(-2, 2, 11)], optn1=True, optn2=False)


if __name__ == '__main__':
    unittest.main()
