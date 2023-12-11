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
class TestEDIBlackBox(unittest.TestCase):
    def test_edi_blackbox_variable(self):
        "Tests the black box variable class"
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import (
            BlackBoxFunctionModel_Variable,
            TypeCheckedList,
            BBList,
            BlackBoxFunctionModel,
        )

        x = BlackBoxFunctionModel_Variable('x', '')
        x_print = x.__repr__()
        x_name = x.name
        x_units = x.units
        x_size = x.size
        x_desc = x.description
        self.assertRaises(ValueError, x.__init__, *(1.0, ''))

        x.__init__('x', units.dimensionless)

        self.assertRaises(ValueError, x.__init__, *('x', 1.0))

        x.__init__('x', units.dimensionless, '', 'flex')
        x.__init__('x', units.dimensionless, '', ['flex', 2])
        x.__init__('x', units.dimensionless, '', 2)
        x.__init__('x', units.dimensionless, '', [2, 2])

        self.assertRaises(ValueError, x.__init__, *('x', '', '', [[], 2]))
        self.assertRaises(ValueError, x.__init__, *('x', '', '', [2, 1]))

        x.__init__('x', units.dimensionless, '', None)
        x.__init__('x', None, '', None)

        self.assertRaises(ValueError, x.__init__, *('x', '', '', 1))
        self.assertRaises(ValueError, x.__init__, *('x', '', '', {}))
        self.assertRaises(ValueError, x.__init__, *('x', '', 1.0))

    def test_edi_blackbox_tcl(self):
        "Tests the black box type checked list class"
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import (
            BlackBoxFunctionModel_Variable,
            TypeCheckedList,
            BBList,
            BlackBoxFunctionModel,
        )

        tcl = TypeCheckedList(int, [1, 2, 3])
        tcl[1] = 1
        tcl[0:2] = [1, 2]

        self.assertRaises(ValueError, tcl.__init__, *(int, 1))
        self.assertRaises(ValueError, tcl.__setitem__, *(1, 3.333))
        self.assertRaises(ValueError, tcl.__setitem__, *(1, [1, 2.222]))
        self.assertRaises(ValueError, tcl.append, *(2.222,))

    def test_edi_blackbox_bbl(self):
        "Tests the black box BBList class"
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import (
            BlackBoxFunctionModel_Variable,
            TypeCheckedList,
            BBList,
            BlackBoxFunctionModel,
        )

        bbl = BBList()
        bbl.append('x', '')
        bbl.append('y', '')
        bbl.append(BlackBoxFunctionModel_Variable('z', ''))
        bbl.append(var=BlackBoxFunctionModel_Variable('u', ''))
        self.assertRaises(
            ValueError, bbl.append, *(BlackBoxFunctionModel_Variable('u', ''),)
        )
        self.assertRaises(ValueError, bbl.append, *('badvar',))
        self.assertRaises(ValueError, bbl.append, *(2.222,))

        self.assertRaises(ValueError, bbl.append, *('bv', '', ''), **{'units': 'm'})
        self.assertRaises(ValueError, bbl.append, **{'units': 'm', 'description': 'hi'})
        self.assertRaises(ValueError, bbl.append, **{'name': 'x', 'units': ''})
        self.assertRaises(ValueError, bbl.append, *('bv', '', '', 0, 'extra'))

        xv = bbl['x']
        xv2 = bbl[0]
        self.assertRaises(ValueError, bbl.__getitem__, *(2.22,))

    def test_edi_blackbox_someexceptions(self):
        "Tests some of the exceptions in the black box model class"
        import numpy as np
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import (
            BlackBoxFunctionModel_Variable,
            TypeCheckedList,
            BBList,
            BlackBoxFunctionModel,
        )

        bb = BlackBoxFunctionModel()
        bb.inputVariables_optimization = [1, 2, 3]
        # bb.set_input_values(np.array([1,2,3]))
        self.assertRaises(ValueError, bb.input_names, *())
        # self.assertRaises(ValueError,bb.fillCache,*( ))

        bb = BlackBoxFunctionModel()
        bb.outputVariables_optimization = [1, 2, 3]
        self.assertRaises(ValueError, bb.output_names, *())

    def test_edi_blackbox_etc_1(self):
        "Tests a black box assertion issue"
        from pyomo.contrib.edi.blackBoxFunctionModel import (
            BlackBoxFunctionModel_Variable,
            TypeCheckedList,
            BBList,
            BlackBoxFunctionModel,
        )

        bbfm = BlackBoxFunctionModel()
        self.assertRaises(AttributeError, bbfm.BlackBox, ())

    def test_edi_blackbox_etc_2(self):
        "Tests a black box assertion issue"
        import numpy as np
        from pyomo.environ import units
        import pyomo.environ as pyo
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import (
            BlackBoxFunctionModel_Variable,
            TypeCheckedList,
            BBList,
            BlackBoxFunctionModel,
        )

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

        f.__dict__['constraint_2'].get_external_model().set_input_values(
            np.array([2.0, 2.0])
        )
        f.__dict__['constraint_2'].get_external_model().inputVariables_optimization = [
            1,
            2,
        ]
        # f.__dict__['constraint_2'].get_external_model().fillCache()
        self.assertRaises(
            ValueError, f.__dict__['constraint_2'].get_external_model().fillCache, *()
        )

    def test_edi_blackbox_etc_3(self):
        "Tests a black box assertion issue"
        import numpy as np
        from pyomo.environ import units
        import pyomo.environ as pyo
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import (
            BlackBoxFunctionModel_Variable,
            TypeCheckedList,
            BBList,
            BlackBoxFunctionModel,
        )

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
                return ['err'], [[dzdx, dzdy]]  # return z, grad(z), hess(z)...

        f.ConstraintList([z <= 1 * units.m**2, [z, '==', [x, y], UnitCircle()]])

        self.assertRaises(
            ValueError, f.__dict__['constraint_2'].get_external_model().fillCache, *()
        )

    def test_edi_blackbox_etc_4(self):
        "Tests a black box assertion issue"
        import numpy as np
        from pyomo.environ import units
        import pyomo.environ as pyo
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import (
            BlackBoxFunctionModel_Variable,
            TypeCheckedList,
            BBList,
            BlackBoxFunctionModel,
        )

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

        f.__dict__['constraint_2'].get_external_model().set_input_values(
            np.array([2.0, 2.0])
        )
        f.__dict__['constraint_2'].get_external_model().outputVariables_optimization = [
            1
        ]
        # f.__dict__['constraint_2'].get_external_model().fillCache()
        self.assertRaises(
            ValueError, f.__dict__['constraint_2'].get_external_model().fillCache, *()
        )

    def test_edi_blackbox_example_1(self):
        "Tests a black box example construction"
        from pyomo.environ import units
        import pyomo.environ as pyo
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import (
            BlackBoxFunctionModel_Variable,
            TypeCheckedList,
            BBList,
            BlackBoxFunctionModel,
        )

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

        f.__dict__['constraint_2'].get_external_model().set_input_values(
            np.array([2.0, 2.0])
        )
        opt = f.__dict__['constraint_2'].get_external_model().evaluate_outputs()
        jac = (
            f.__dict__['constraint_2']
            .get_external_model()
            .evaluate_jacobian_outputs()
            .todense()
        )

        self.assertAlmostEqual(opt[0], 8)
        self.assertAlmostEqual(jac[0, 0], 4)
        self.assertAlmostEqual(jac[0, 1], 4)

        sm = f.__dict__['constraint_2'].get_external_model().summary
        e_print = f.__dict__['constraint_2'].get_external_model().__repr__()

    def test_edi_blackbox_example_2(self):
        "Tests a black box example construction"
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

            def BlackBox(self, *args, **kwargs):  # The actual function that does things
                runCases, remainingKwargs = self.parseInputs(
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

        f.__dict__['constraint_1'].get_external_model().set_input_values(np.ones(3) * 2)
        opt = f.__dict__['constraint_1'].get_external_model().evaluate_outputs()
        jac = (
            f.__dict__['constraint_1']
            .get_external_model()
            .evaluate_jacobian_outputs()
            .todense()
        )

        self.assertAlmostEqual(opt[0], 4)
        self.assertAlmostEqual(opt[1], 4)
        self.assertAlmostEqual(opt[2], 4)
        self.assertAlmostEqual(jac[0, 0], 4)
        self.assertAlmostEqual(jac[0, 1], 0)
        self.assertAlmostEqual(jac[0, 2], 0)
        self.assertAlmostEqual(jac[0, 1], 0)
        self.assertAlmostEqual(jac[1, 1], 4)
        self.assertAlmostEqual(jac[2, 1], 0)
        self.assertAlmostEqual(jac[2, 0], 0)
        self.assertAlmostEqual(jac[2, 1], 0)
        self.assertAlmostEqual(jac[2, 2], 4)

        sm = f.__dict__['constraint_1'].get_external_model().summary
        e_print = f.__dict__['constraint_1'].get_external_model().__repr__()

    def test_edi_blackbox_example_3(self):
        "Tests a black box example construction"
        import numpy as np
        from pyomo.environ import units
        import pyomo.environ as pyo
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import (
            BlackBoxFunctionModel_Variable,
            TypeCheckedList,
            BBList,
            BlackBoxFunctionModel,
        )

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='', description='x variable', size=3)
        y = f.Variable(name='y', guess=1.0, units='', description='y variable')
        f.Objective(y)

        class Norm_2(BlackBoxFunctionModel):
            def __init__(self):
                super().__init__()
                self.description = 'This model evaluates the two norm'
                self.inputs.append(
                    name='x', units='', description='The x variable', size=3
                )
                self.outputs.append(name='y', units='', description='The y variable')
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, *args, **kwargs):  # The actual function that does things
                runCases, remainingKwargs = self.parseInputs(
                    *args, **kwargs
                )

                x = self.sanitizeInputs(runCases[0]['x'])
                x = np.array([pyo.value(xval) for xval in x], dtype=np.float64)

                y = x[0] ** 2 + x[1] ** 2 + x[2] ** 2  # Compute y
                dydx0 = 2 * x[0]  # Compute dy/dx0
                dydx1 = 2 * x[1]  # Compute dy/dx1
                dydx2 = 2 * x[2]  # Compute dy/dx2

                y = y * units.dimensionless
                dydx = np.array([dydx0, dydx1, dydx2]) * units.dimensionless
                # dydx0 = dydx0 * units.dimensionless
                # dydx1 = dydx1 * units.dimensionless
                # dydx2 = dydx2 * units.dimensionless

                return y, [dydx]  # return z, grad(z), hess(z)...

        f.ConstraintList(
            [{'outputs': y, 'operators': '==', 'inputs': x, 'black_box': Norm_2()}]
        )

        f.__dict__['constraint_1'].get_external_model().set_input_values(np.ones(3) * 2)
        opt = f.__dict__['constraint_1'].get_external_model().evaluate_outputs()
        jac = (
            f.__dict__['constraint_1']
            .get_external_model()
            .evaluate_jacobian_outputs()
            .todense()
        )

        self.assertAlmostEqual(opt[0], 12)
        self.assertAlmostEqual(jac[0, 0], 4)
        self.assertAlmostEqual(jac[0, 1], 4)
        self.assertAlmostEqual(jac[0, 2], 4)

        sm = f.__dict__['constraint_1'].get_external_model().summary
        e_print = f.__dict__['constraint_1'].get_external_model().__repr__()

    def test_edi_blackbox_example_4(self):
        "Tests a black box example construction"
        import numpy as np
        from pyomo.environ import units
        import pyomo.environ as pyo
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import (
            BlackBoxFunctionModel_Variable,
            TypeCheckedList,
            BBList,
            BlackBoxFunctionModel,
        )

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='', description='x variable')
        y = f.Variable(name='y', guess=1.0, units='', description='y variable', size=3)
        f.Objective(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)

        class VectorCast(BlackBoxFunctionModel):
            def __init__(self):
                super().__init__()
                self.description = 'This model evaluates the two norm'
                self.inputs.append(name='x', units='', description='The x variable')
                self.outputs.append(
                    name='y', units='', description='The y variable', size=3
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, *args, **kwargs):  # The actual function that does things
                runCases, remainingKwargs = self.parseInputs(
                    *args, **kwargs
                )

                x = pyo.value(self.sanitizeInputs(runCases[0]['x']))

                y = np.array([x, x, x]) * units.dimensionless
                dydx = np.array([1.0, 1.0, 1.0]) * units.dimensionless

                return y, [dydx]  # return z, grad(z), hess(z)...

        f.ConstraintList(
            [{'outputs': y, 'operators': '==', 'inputs': x, 'black_box': VectorCast()}]
        )

        f.__dict__['constraint_1'].get_external_model().set_input_values(np.ones(3) * 2)
        opt = f.__dict__['constraint_1'].get_external_model().evaluate_outputs()
        jac = (
            f.__dict__['constraint_1']
            .get_external_model()
            .evaluate_jacobian_outputs()
            .todense()
        )

        self.assertAlmostEqual(opt[0], 2.0)
        self.assertAlmostEqual(opt[1], 2.0)
        self.assertAlmostEqual(opt[2], 2.0)
        self.assertAlmostEqual(jac[0, 0], 1.0)
        self.assertAlmostEqual(jac[1, 0], 1.0)
        self.assertAlmostEqual(jac[2, 0], 1.0)

        sm = f.__dict__['constraint_1'].get_external_model().summary
        e_print = f.__dict__['constraint_1'].get_external_model().__repr__()

    def test_edi_blackbox_badexample_1(self):
        "Tests a black box example construction"
        import numpy as np
        from pyomo.environ import units
        import pyomo.environ as pyo
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import (
            BlackBoxFunctionModel_Variable,
            TypeCheckedList,
            BBList,
            BlackBoxFunctionModel,
        )

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='', description='x variable', size=3)
        y = f.Variable(name='y', guess=1.0, units='', description='y variable')
        f.Objective(y)

        class Norm_2(BlackBoxFunctionModel):
            def __init__(self):
                super().__init__()
                self.description = 'This model evaluates the two norm'
                self.inputs.append(
                    name='x', units='', description='The x variable', size=3
                )
                self.outputs.append(name='y', units='', description='The y variable')
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, *args, **kwargs):  # The actual function that does things
                runCases, remainingKwargs = self.parseInputs(
                    *args, **kwargs
                )

                x = self.sanitizeInputs(runCases[0]['x'])
                x = np.array([pyo.value(xval) for xval in x], dtype=np.float64)

                y = x[0] ** 2 + x[1] ** 2 + x[2] ** 2  # Compute y
                dydx0 = 2 * x[0]  # Compute dy/dx0
                dydx1 = 2 * x[1]  # Compute dy/dx1
                dydx2 = 2 * x[2]  # Compute dy/dx2

                y = y * units.dimensionless
                # dydx = np.array([dydx0,dydx1,dydx2]) * units.dimensionless
                dydx0 = dydx0 * units.dimensionless
                dydx1 = dydx1 * units.dimensionless
                dydx2 = dydx2 * units.dimensionless

                return y, [[dydx0, dydx1, dydx2]]  # return z, grad(z), hess(z)...

        f.ConstraintList(
            [{'outputs': y, 'operators': '==', 'inputs': x, 'black_box': Norm_2()}]
        )

        f.__dict__['constraint_1'].get_external_model().set_input_values(np.ones(3) * 2)
        self.assertRaises(
            ValueError,
            f.__dict__['constraint_1'].get_external_model().evaluate_outputs,
            *()
        )

    def test_edi_blackbox_smallfunctions(self):
        "Tests the more general value and convert functions"
        import numpy as np
        from pyomo.environ import units
        import pyomo.environ as pyo
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import (
            BlackBoxFunctionModel_Variable,
            TypeCheckedList,
            BBList,
            BlackBoxFunctionModel,
        )

        bb = BlackBoxFunctionModel()
        t1 = bb.convert(2 * units.m, units.ft)
        t2 = bb.convert(np.ones([2, 2]) * units.m, units.ft)
        self.assertRaises(ValueError, bb.convert, *('err', units.ft))

        t3 = bb.pyomo_value(2 * units.m)
        t3 = bb.pyomo_value(np.ones([2, 2]) * units.m)
        t4 = bb.pyomo_value(np.ones([2, 2]))

        bb.sizeCheck([2, 2], np.ones([2, 2]) * units.m)
        self.assertRaises(ValueError, bb.sizeCheck, *(2, 3 * units.m))
        self.assertRaises(
            ValueError, bb.sizeCheck, *([10, 10, 10], np.ones([2, 2]) * units.m)
        )
        self.assertRaises(
            ValueError, bb.sizeCheck, *([10, 10], np.ones([2, 2]) * units.m)
        )
        self.assertRaises(ValueError, bb.sizeCheck, *([10, 10], []))

    # def test_edi_blackbox_bare_example_1(self):
    #     "Tests a black box example construction without an optimization problem"
    #     import numpy as np
    #     import pyomo.environ as pyo
    #     from pyomo.environ import units
    #     from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

    #     class SignomialTest(BlackBoxFunctionModel):
    #         def __init__(self):
    #             # Set up all the attributes by calling Model.__init__
    #             super().__init__()

    #             # Setup Inputs
    #             self.inputs.append('x', '', 'Independent Variable')

    #             # Setup Outputs
    #             self.outputs.append('y', '', 'Dependent Variable')

    #             # Simple model description
    #             self.description = (
    #                 'This model evaluates the function: max([-6*x-6, x**4-3*x**2])'
    #             )

    #             self.availableDerivative = 1

    #         # standard function call is y(, dydx, ...) = self.BlackBox(**{'x1':x1, 'x2':x2, ...})
    #         def BlackBox(self, *args, **kwargs):
    #             runCases, returnMode, extras = self.parseInputs(*args, **kwargs)
    #             x = np.array(
    #                 [pyo.value(runCases[i]['x']) for i in range(0, len(runCases))]
    #             )

    #             y = np.maximum(-6 * x - 6, x**4 - 3 * x**2)
    #             dydx = 4 * x**3 - 6 * x
    #             ddy_ddx = 12 * x**2 - 6
    #             gradientSwitch = -6 * x - 6 > x**4 - 3 * x**2
    #             dydx[gradientSwitch] = -6
    #             ddy_ddx[gradientSwitch] = 0

    #             y = [yval * units.dimensionless for yval in y]
    #             dydx = [dydx[i] * units.dimensionless for i in range(0, len(dydx))]

    #             if returnMode < 0:
    #                 returnMode = -1 * (returnMode + 1)
    #                 if returnMode == 0:
    #                     return y[0]
    #                 if returnMode == 1:
    #                     return y[0], dydx
    #             else:
    #                 if returnMode == 0:
    #                     opt = []
    #                     for i in range(0, len(y)):
    #                         opt.append([y[i]])
    #                     return opt
    #                 if returnMode == 1:
    #                     opt = []
    #                     for i in range(0, len(y)):
    #                         opt.append([[y[i]], [[[dydx[i]]]]])
    #                     return opt

    #     s = SignomialTest()
    #     ivals = [[x] for x in np.linspace(-2, 2, 11)]

    #     # How the black box may be called using EDI
    #     bbo = s.BlackBox(**{'x': 0.5})
    #     bbo = s.BlackBox({'x': 0.5})
    #     bbo = s.BlackBox(**{'x': 0.5, 'optn': True})

    #     # Additional options available with parseInputs
    #     bbo = s.BlackBox(*[0.5], **{'optn1': True, 'optn2': False})
    #     bbo = s.BlackBox(*[0.5, True], **{'optn': False})
    #     bbo = s.BlackBox({'x': [x for x in np.linspace(-2, 2, 11)]})
    #     bbo = s.BlackBox([{'x': x} for x in np.linspace(-2, 2, 11)])
    #     bbo = s.BlackBox([[x] for x in np.linspace(-2, 2, 11)])
    #     bbo = s.BlackBox([[x] for x in np.linspace(-2, 2, 11)], True, optn=False)
    #     bbo = s.BlackBox([[x] for x in np.linspace(-2, 2, 11)], optn1=True, optn2=False)

    #     sm = s.summary

    #     self.assertRaises(ValueError, s.BlackBox, *([[[1, 2], 2, 3], [1, 2, 3]]))
    #     self.assertRaises(ValueError, s.BlackBox, *([[np.ones(2), 2, 3], [1, 2, 3]]))

    #     self.assertRaises(ValueError, s.sanitizeInputs, *())
    #     self.assertRaises(ValueError, s.sanitizeInputs, *(1, 2, 3))
    #     self.assertRaises(
    #         ValueError, s.sanitizeInputs, **{'z': 2.0 * units.dimensionless}
    #     )
    #     self.assertRaises(ValueError, s.sanitizeInputs, *(2.0 * units.ft,))
    #     self.assertRaises(NotImplementedError, s.checkOutputs, *())

    # def test_edi_blackbox_bare_example_2(self):
    #     "Tests a black box example construction without an optimization problem"
    #     import numpy as np
    #     import pyomo.environ as pyo
    #     from pyomo.environ import units
    #     from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel
    #     from pyomo.common.formatting import tostr

    #     class PassThrough(BlackBoxFunctionModel):
    #         def __init__(self):
    #             # Set up all the attributes by calling Model.__init__
    #             super().__init__()

    #             # Setup Inputs
    #             self.inputs.append('x', '', 'Independent Variable', size=[2, 2])

    #             # Setup Outputs
    #             self.outputs.append('y', '', 'Dependent Variable', size=[2, 2])

    #             # Simple model description
    #             self.description = 'This model is a pass through)'

    #             self.availableDerivative = 1

    #         # standard function call is y(, dydx, ...) = self.BlackBox(**{'x1':x1, 'x2':x2, ...})
    #         def BlackBox(self, *args, **kwargs):
    #             runCases, returnMode, extras = self.parseInputs(*args, **kwargs)
    #             x = [
    #                 self.pyomo_value(runCases[i]['x']) for i in range(0, len(runCases))
    #             ]

    #             y = []
    #             dydx = []

    #             for xval in x:
    #                 y.append(xval * units.dimensionless)
    #                 dydx_temp = np.zeros([2, 2, 2, 2])
    #                 dydx_temp[0, 0, 0, 0] = 1.0
    #                 dydx_temp[0, 1, 0, 1] = 1.0
    #                 dydx_temp[1, 0, 1, 0] = 1.0
    #                 dydx_temp[1, 1, 1, 1] = 1.0

    #                 dydx.append(dydx_temp * units.dimensionless)

    #             if returnMode < 0:
    #                 returnMode = -1 * (returnMode + 1)
    #                 if returnMode == 0:
    #                     return y[0]
    #                 if returnMode == 1:
    #                     return y[0], dydx
    #             else:
    #                 if returnMode == 0:
    #                     opt = []
    #                     for i in range(0, len(y)):
    #                         opt.append([y[i]])
    #                     return opt
    #                 if returnMode == 1:
    #                     opt = []
    #                     for i in range(0, len(y)):
    #                         opt.append([[y[i]], [[[dydx[i]]]]])
    #                     return opt

    #     bb = PassThrough()
    #     ivals = [
    #         [np.eye(2) * units.dimensionless],
    #         [np.ones([2, 2]) * units.dimensionless],
    #         [np.zeros([2, 2]) * units.dimensionless],
    #     ]

    #     xv = np.eye(2) * units.dimensionless

    #     # How the black box may be called using EDI
    #     bbo = bb.BlackBox(**{'x': xv})
    #     bbo = bb.BlackBox({'x': xv})
    #     bbo = bb.BlackBox(**{'x': xv, 'optn': True})

    #     # # Additional options available with parseInputs
    #     bbo = bb.BlackBox(*[xv], **{'optn1': True, 'optn2': False})
    #     bbo = bb.BlackBox(*[xv, True], **{'optn': False})
    #     bbo = bb.BlackBox({'x': [x[0] for x in ivals]})
    #     bbo = bb.BlackBox([{'x': x[0]} for x in ivals])
    #     bbo = bb.BlackBox([[x[0]] for x in ivals])
    #     bbo = bb.BlackBox([[x[0]] for x in ivals], True, optn=False)
    #     bbo = bb.BlackBox([[x[0]] for x in ivals], optn1=True, optn2=False)

    #     sm = bb.summary

    #     self.assertRaises(ValueError, bb.sanitizeInputs, *(np.ones([2, 2]) * units.ft,))

    # def test_edi_blackbox_bare_example_3(self):
    #     "Tests a black box example construction"
    #     import numpy as np
    #     from pyomo.environ import units
    #     import pyomo.environ as pyo
    #     from pyomo.contrib.edi import Formulation
    #     from pyomo.contrib.edi.blackBoxFunctionModel import (
    #         BlackBoxFunctionModel_Variable,
    #         TypeCheckedList,
    #         BBList,
    #         BlackBoxFunctionModel,
    #     )

    #     class Norm_2(BlackBoxFunctionModel):
    #         def __init__(self):
    #             super().__init__()
    #             self.description = 'This model evaluates the two norm'
    #             self.inputs.append(
    #                 name='x', units='', description='The x variable', size=3
    #             )
    #             self.inputs.append(
    #                 name='y', units='', description='The y variable', size=2
    #             )
    #             self.outputs.append(name='z', units='', description='The z variable')
    #             self.availableDerivative = 1
    #             self.post_init_setup(len(self.inputs))

    #         def BlackBox(self, *args, **kwargs):  # The actual function that does things
    #             runCases, returnMode, remainingKwargs = self.parseInputs(
    #                 *args, **kwargs
    #             )

    #             x = [rc['x'] for rc in runCases][0]
    #             x = np.array([self.pyomo_value(xval) for xval in x], dtype=np.float64)

    #             y = [rc['y'] for rc in runCases][0]
    #             y = np.array([self.pyomo_value(yval) for yval in y], dtype=np.float64)

    #             z = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + y[0] ** 2 + y[1] ** 2
    #             dzdx0 = 2 * x[0]  # Compute dy/dx0
    #             dzdx1 = 2 * x[1]  # Compute dy/dx1
    #             dzdx2 = 2 * x[2]  # Compute dy/dx2
    #             dzdy0 = 2 * y[0]
    #             dzdy1 = 2 * y[1]

    #             z = z * units.dimensionless
    #             dz = np.array([dzdx0, dzdx1, dzdx2, dzdy0, dzdy1]) * units.dimensionless
    #             # dydx0 = dydx0 * units.dimensionless
    #             # dydx1 = dydx1 * units.dimensionless
    #             # dydx2 = dydx2 * units.dimensionless

    #             return z, [dz]  # return z, grad(z), hess(z)...

    #     bb = Norm_2()
    #     bbo = bb.BlackBox(
    #         {
    #             'x': np.array([0, 0, 0]) * units.dimensionless,
    #             'y': np.array([0, 0]) * units.dimensionless,
    #         }
    #     )
    #     bbo = bb.BlackBox(
    #         np.array([0, 0, 0]) * units.dimensionless,
    #         y=np.array([0, 0]) * units.dimensionless,
    #     )

    #     self.assertRaises(
    #         ValueError,
    #         bb.BlackBox,
    #         *(
    #             {
    #                 'er': np.array([0, 0, 0]) * units.dimensionless,
    #                 'y': np.array([0, 0]) * units.dimensionless,
    #             },
    #         )
    #     )
    #     self.assertRaises(ValueError, bb.BlackBox, *('err',))
    #     self.assertRaises(
    #         ValueError,
    #         bb.BlackBox,
    #         *(
    #             np.array([0, 0, 0]) * units.dimensionless,
    #             np.array([0, 0]) * units.dimensionless,
    #         ),
    #         **{'x': 'err too many'}
    #     )
    #     self.assertRaises(
    #         ValueError,
    #         bb.BlackBox,
    #         *(np.array([0, 0, 0]) * units.dimensionless,),
    #         **{'notY': np.array([0, 0]) * units.dimensionless}
    #     )

    # def test_edi_blackbox_bare_example_4(self):
    #     "Tests a black box example construction without an optimization problem"
    #     import numpy as np
    #     import pyomo.environ as pyo
    #     from pyomo.environ import units
    #     from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel
    #     from pyomo.common.formatting import tostr

    #     class PassThrough(BlackBoxFunctionModel):
    #         def __init__(self):
    #             # Set up all the attributes by calling Model.__init__
    #             super().__init__()

    #             # Setup Inputs
    #             self.inputs.append('x', '', 'X Variable')
    #             self.inputs.append('y', '', 'Y Variable')

    #             # Setup Outputs
    #             self.outputs.append('u', '', 'U Variable')
    #             self.outputs.append('v', '', 'V Variable')

    #             # Simple model description
    #             self.description = 'This model is a pass through)'

    #             self.availableDerivative = 1

    #         # standard function call is y(, dydx, ...) = self.BlackBox(**{'x1':x1, 'x2':x2, ...})
    #         def BlackBox(self, *args, **kwargs):
    #             runCases, returnMode, extras = self.parseInputs(*args, **kwargs)
    #             x = [
    #                 self.pyomo_value(runCases[i]['x']) for i in range(0, len(runCases))
    #             ]
    #             y = [
    #                 self.pyomo_value(runCases[i]['x']) for i in range(0, len(runCases))
    #             ]

    #             u = []
    #             dudx = []
    #             dudy = []
    #             v = []
    #             dvdx = []
    #             dvdy = []

    #             for xval in x:
    #                 u.append(xval * units.dimensionless)
    #                 dudx.append(1.0 * units.dimensionless)
    #                 dudy.append(0.0 * units.dimensionless)

    #             for yval in y:
    #                 v.append(yval * units.dimensionless)
    #                 dvdx.append(0.0 * units.dimensionless)
    #                 dvdy.append(1.0 * units.dimensionless)

    #             if returnMode < 0:
    #                 returnMode = -1 * (returnMode + 1)
    #                 if returnMode == 0:
    #                     return [u[0], v[0]]
    #                 if returnMode == 1:
    #                     return [u[0], v[0]], [[dudx[0], dudy[0]], [dvdx[0], dvdy[0]]]
    #             else:
    #                 if returnMode == 0:
    #                     opt = []
    #                     for i in range(0, len(y)):
    #                         opt.append([u[i], v[i]])
    #                     return opt
    #                 if returnMode == 1:
    #                     opt = []
    #                     for i in range(0, len(y)):
    #                         opt.append(
    #                             [[u[i], v[i]], [[dudx[i], dudy[i]], [dvdx[i], dvdy[i]]]]
    #                         )
    #                     return opt

    #     bb = PassThrough()
    #     bbo = bb.BlackBox(1.0, 1.0)
    #     bbo = bb.BlackBox({'x': np.linspace(0, 10, 11), 'y': np.linspace(0, 10, 11)})


if __name__ == '__main__':
    unittest.main()
