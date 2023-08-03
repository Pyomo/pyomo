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

np, numpy_available = attempt_import(
    'numpy', 'edi requires numpy'
)
# scipy, scipy_available = attempt_import(
#     'scipy', 'inverse_reduced_hessian requires scipy'
# )

# if not (numpy_available and scipy_available):
if not numpy_available:
    raise unittest.SkipTest(
        'edi.formulation tests require numpy'
    )

@unittest.skipIf(not pint_available, 'Testing units requires pint')
class TestEDIBlackBox(unittest.TestCase):
    def test_edi_blackbox_variable(self):
        "Tests the black box variable class"
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import BlackBoxFunctionModel_Variable, TypeCheckedList, BBList, BlackBoxFunctionModel

        x = BlackBoxFunctionModel_Variable('x','')
        x_print = x.__repr__()
        x_name = x.name
        x_units = x.units 
        x_size = x.size 
        x_desc = x.description
        self.assertRaises(ValueError,x.__init__,*(1.0, ''))

        x.__init__('x', units.dimensionless)

        self.assertRaises(ValueError,x.__init__,*('x', 1.0))        

        x.__init__('x', units.dimensionless, '', 'flex')
        x.__init__('x', units.dimensionless, '', ['flex',2])
        x.__init__('x', units.dimensionless, '', 2)
        x.__init__('x', units.dimensionless, '', [2,2])

        self.assertRaises(ValueError,x.__init__,*('x', '','',[[],2]))   
        self.assertRaises(ValueError,x.__init__,*('x', '','',[2,1]))        

        x.__init__('x', units.dimensionless, '', None)
        self.assertRaises(ValueError,x.__init__,*('x', '','', 1 ))
        self.assertRaises(ValueError,x.__init__,*('x', '','', {} ))    
        self.assertRaises(ValueError,x.__init__,*('x', '', 1.0 ))        




    def test_edi_blackbox_tcl(self):
        "Tests the black box type checked list class"
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import BlackBoxFunctionModel_Variable, TypeCheckedList, BBList, BlackBoxFunctionModel

        tcl = TypeCheckedList(int, [1,2,3])
        tcl[1] = 1
        tcl[0:2] = [1,2]
        
        self.assertRaises(ValueError,tcl.__init__,*( int, 1 ))
        self.assertRaises(ValueError,tcl.__setitem__,*( 1, 3.333 ))
        self.assertRaises(ValueError,tcl.__setitem__,*( 1, [1, 2.222] ))
        self.assertRaises(ValueError,tcl.append,*(  2.222 ,))



    def test_edi_blackbox_bbl(self):
        "Tests the black box BBList class"
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import BlackBoxFunctionModel_Variable, TypeCheckedList, BBList, BlackBoxFunctionModel

        bbl = BBList()
        bbl.append('x','')
        bbl.append('y','')
        bbl.append(BlackBoxFunctionModel_Variable('z',''))
        bbl.append(var = BlackBoxFunctionModel_Variable('u',''))
        self.assertRaises(ValueError,bbl.append,*( BlackBoxFunctionModel_Variable('u',''),))
        self.assertRaises(ValueError,bbl.append,*( 'badvar',))
        self.assertRaises(ValueError,bbl.append,*( 2.222,))

        self.assertRaises(ValueError,bbl.append,*( 'bv','',''),**{'units':'m'})
        self.assertRaises(ValueError,bbl.append,**{'units':'m','description':'hi'})
        self.assertRaises(ValueError,bbl.append,**{'name':'x', 'units':''})
        self.assertRaises(ValueError,bbl.append,*( 'bv','','',0,'extra'))

        xv = bbl['x']
        xv2 = bbl[0]
        self.assertRaises(ValueError,bbl.__getitem__,*( 2.22, ))


    def test_edi_blackbox_someexceptions(self):
        "Tests some of the exceptions in the black box model class"
        import numpy as np
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import BlackBoxFunctionModel_Variable, TypeCheckedList, BBList, BlackBoxFunctionModel

        bb = BlackBoxFunctionModel()
        bb.inputVariables_optimization = [1,2,3]
        # bb.set_input_values(np.array([1,2,3]))
        self.assertRaises(ValueError,bb.input_names,*( ))
        # self.assertRaises(ValueError,bb.fillCache,*( ))


        bb = BlackBoxFunctionModel()
        bb.outputVariables_optimization = [1,2,3]
        self.assertRaises(ValueError,bb.output_names,*( ))


    def test_edi_blackbox_etc_1(self):
        "Tests a black box assertion issue"
        from pyomo.contrib.edi.blackBoxFunctionModel import BlackBoxFunctionModel_Variable, TypeCheckedList, BBList, BlackBoxFunctionModel
        bbfm = BlackBoxFunctionModel()
        self.assertRaises(AttributeError,bbfm.BlackBox,())


    def test_edi_blackbox_etc_2(self):
        "Tests a black box assertion issue"
        import numpy as np
        from pyomo.environ import units
        import pyomo.environ as pyo
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import BlackBoxFunctionModel_Variable, TypeCheckedList, BBList, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'x variable')
        y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'y variable')
        z = f.Variable(name = 'z', guess = 1.0, units = 'm^2', description = 'Output var')
        f.Objective( x + y )
        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super().__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append( name = 'x',
                                    units = 'ft' ,
                                    description = 'The x variable' )
                self.inputs.append( name = 'y',
                                    units = 'ft' ,
                                    description = 'The y variable' )
                self.outputs.append(name = 'z',
                                    units = 'ft**2',
                                    description = 'Output variable' )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))
            def BlackBox(self, x, y): # The actual function that does things
                # Converts to correct units then casts to float
                x = pyo.value(units.convert(x,self.inputs[0].units))
                y = pyo.value(units.convert(y,self.inputs[1].units))
                z = x**2 + y**2 # Compute z
                dzdx = 2*x      # Compute dz/dx
                dzdy = 2*y      # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft # units.ft**2 / units.ft
                dzdy *= units.ft # units.ft**2 / units.ft
                return z, [dzdx, dzdy] # return z, grad(z), hess(z)...


        f.ConstraintList(
            [
                z <= 1*units.m**2 ,
                [ z, '==', [x,y], UnitCircle() ] ,
            ]
        )

        f.__dict__['constraint_2'].get_external_model().set_input_values(np.array([2.0, 2.0]))
        f.__dict__['constraint_2'].get_external_model().inputVariables_optimization = [1,2]
        # f.__dict__['constraint_2'].get_external_model().fillCache()
        self.assertRaises(ValueError,f.__dict__['constraint_2'].get_external_model().fillCache,*())


    def test_edi_blackbox_etc_3(self):
        "Tests a black box assertion issue"
        import numpy as np
        from pyomo.environ import units
        import pyomo.environ as pyo
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import BlackBoxFunctionModel_Variable, TypeCheckedList, BBList, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'x variable')
        y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'y variable')
        z = f.Variable(name = 'z', guess = 1.0, units = 'm^2', description = 'Output var')
        f.Objective( x + y )
        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super().__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append( name = 'x',
                                    units = 'ft' ,
                                    description = 'The x variable' )
                self.inputs.append( name = 'y',
                                    units = 'ft' ,
                                    description = 'The y variable' )
                self.outputs.append(name = 'z',
                                    units = 'ft**2',
                                    description = 'Output variable' )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))
            def BlackBox(self, x, y): # The actual function that does things
                # Converts to correct units then casts to float
                x = pyo.value(units.convert(x,self.inputs[0].units))
                y = pyo.value(units.convert(y,self.inputs[1].units))
                z = x**2 + y**2 # Compute z
                dzdx = 2*x      # Compute dz/dx
                dzdy = 2*y      # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft # units.ft**2 / units.ft
                dzdy *= units.ft # units.ft**2 / units.ft
                return ['err'], [[dzdx, dzdy]] # return z, grad(z), hess(z)...


        f.ConstraintList(
            [
                z <= 1*units.m**2 ,
                [ z, '==', [x,y], UnitCircle() ] ,
            ]
        )

        self.assertRaises(ValueError,f.__dict__['constraint_2'].get_external_model().fillCache,*())

    def test_edi_blackbox_example_1(self):
        "Tests a black box example construction"
        from pyomo.environ import units
        import pyomo.environ as pyo
        from pyomo.contrib.edi import Formulation
        from pyomo.contrib.edi.blackBoxFunctionModel import BlackBoxFunctionModel_Variable, TypeCheckedList, BBList, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'x variable')
        y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'y variable')
        z = f.Variable(name = 'z', guess = 1.0, units = 'm^2', description = 'Output var')
        f.Objective( x + y )
        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super().__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append( name = 'x',
                                    units = 'ft' ,
                                    description = 'The x variable' )
                self.inputs.append( name = 'y',
                                    units = 'ft' ,
                                    description = 'The y variable' )
                self.outputs.append(name = 'z',
                                    units = 'ft**2',
                                    description = 'Output variable' )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))
            def BlackBox(self, x, y): # The actual function that does things
                # Converts to correct units then casts to float
                x = pyo.value(units.convert(x,self.inputs[0].units))
                y = pyo.value(units.convert(y,self.inputs[1].units))
                z = x**2 + y**2 # Compute z
                dzdx = 2*x      # Compute dz/dx
                dzdy = 2*y      # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft # units.ft**2 / units.ft
                dzdy *= units.ft # units.ft**2 / units.ft
                return z, [dzdx, dzdy] # return z, grad(z), hess(z)...


        f.ConstraintList(
            [
                z <= 1*units.m**2 ,
                [ z, '==', [x,y], UnitCircle() ] ,
            ]
        )

        f.__dict__['constraint_2'].get_external_model().set_input_values(np.array([2.0, 2.0]))
        opt  = f.__dict__['constraint_2'].get_external_model().evaluate_outputs()
        jac  = f.__dict__['constraint_2'].get_external_model().evaluate_jacobian_outputs().todense()

        self.assertAlmostEqual(opt[0],8)
        self.assertAlmostEqual(jac[0,0],4)
        self.assertAlmostEqual(jac[0,1],4)

        sm      = f.__dict__['constraint_2'].get_external_model().summary
        e_print = f.__dict__['constraint_2'].get_external_model().__repr__()




    def test_edi_blackbox_example_2(self):
        "Tests a black box example construction"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel
        f = Formulation()
        x = f.Variable(name = 'x', guess = 1.0, units = 'm'     , description = 'The x variable', size=3)
        y = f.Variable(name = 'y', guess = 1.0, units = 'm**2'  , description = 'The y variable', size=3)
        f.Objective( y[0] + y[1] + y[2] )
        class Parabola(BlackBoxFunctionModel):
            def __init__(self): 
                super(Parabola, self).__init__()
                self.description = 'This model evaluates the function: y = x**2'
                self.inputs.append(name = 'x', size = 3, units = 'ft' , description = 'The x variable')
                self.outputs.append(name = 'y', size = 3, units = 'ft**2' , description = 'The y variable')
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(*args, **kwargs):# The actual function that does things
                args = list(args)
                self = args.pop(0)
                runCases, returnMode, remainingKwargs = self.parseInputs(*args, **kwargs)
               
                x = self.sanitizeInputs(runCases[0]['x'])
                x = np.array([pyo.value(xval) for xval in x], dtype=np.float64)

                y = x**2    # Compute y
                dydx = 2*x  # Compute dy/dx
                
                y = y * units.ft**2
                dydx = np.diag(dydx)
                dydx = dydx * units.ft # units.ft**2 / units.ft
                
                return y, [dydx] # return z, grad(z), hess(z)...
        f.ConstraintList([ {'outputs':y, 'operators':'==', 'inputs':x, 'black_box':Parabola() },])

        f.__dict__['constraint_1'].get_external_model().set_input_values(np.ones(3)*2)
        opt = f.__dict__['constraint_1'].get_external_model().evaluate_outputs()
        jac = f.__dict__['constraint_1'].get_external_model().evaluate_jacobian_outputs().todense()

        self.assertAlmostEqual(opt[0],4)
        self.assertAlmostEqual(opt[1],4)
        self.assertAlmostEqual(opt[2],4)
        self.assertAlmostEqual(jac[0,0],4)
        self.assertAlmostEqual(jac[0,1],0)
        self.assertAlmostEqual(jac[0,2],0)
        self.assertAlmostEqual(jac[0,1],0)
        self.assertAlmostEqual(jac[1,1],4)
        self.assertAlmostEqual(jac[2,1],0)
        self.assertAlmostEqual(jac[2,0],0)
        self.assertAlmostEqual(jac[2,1],0)
        self.assertAlmostEqual(jac[2,2],4)

        sm      = f.__dict__['constraint_1'].get_external_model().summary
        e_print = f.__dict__['constraint_1'].get_external_model().__repr__()



    def test_edi_blackbox_todo2(self):
        "TODO2"
        import numpy as np
        import pyomo.environ as pyo
        from pyomo.environ import units
        from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

        class SignomialTest(BlackBoxFunctionModel):
            def __init__(self):
                # Set up all the attributes by calling Model.__init__
                super().__init__()

                #Setup Inputs
                self.inputs.append( 'x', '', 'Independent Variable')

                #Setup Outputs
                self.outputs.append( 'y', '', 'Dependent Variable')

                #Simple model description
                self.description = 'This model evaluates the function: max([-6*x-6, x**4-3*x**2])'

                self.availableDerivative = 1

            #standard function call is y(, dydx, ...) = self.BlackBox(**{'x1':x1, 'x2':x2, ...})
            def BlackBox(*args, **kwargs):
                args = list(args)       # convert tuple to list
                self = args.pop(0)      # pop off the self argument
                runCases, returnMode, extras = self.parseInputs(*args, **kwargs)
                x = np.array([ pyo.value(runCases[i]['x']) for i in range(0,len(runCases)) ])

                y = np.maximum(-6*x-6, x**4-3*x**2)
                dydx = 4*x**3 - 6*x
                ddy_ddx = 12*x**2 - 6
                gradientSwitch = -6*x-6 > x**4-3*x**2
                dydx[gradientSwitch] = -6
                ddy_ddx[gradientSwitch] = 0

                y = [ self.checkOutputs(yval) for yval in y ]
                dydx = [dydx[i] * units.dimensionless for i in range(0,len(dydx))]

                if returnMode < 0:
                    returnMode = -1*(returnMode + 1)
                    if returnMode == 0:
                        return y[0]
                    if returnMode == 1:
                        return y[0], dydx
                else:
                    if returnMode == 0:
                        opt = []
                        for i in range(0,len(y)):
                            opt.append([ y[i] ])
                        return opt
                    if returnMode == 1:
                        opt = []
                        for i in range(0,len(y)):
                            opt.append([ [y[i]], [ [[dydx[i]]] ] ])
                        return opt

        s = SignomialTest()
        ivals = [[x] for x in np.linspace(-2,2,11)]

        # How the black box may be called using EDI
        bbo = s.BlackBox(**{'x':0.5})
        bbo = s.BlackBox({'x':0.5})
        bbo = s.BlackBox(**{'x':0.5, 'optn':True})

        # Additional options available with parseInputs
        bbo = s.BlackBox(*[0.5], **{'optn1': True, 'optn2': False})
        bbo = s.BlackBox(*[0.5,True], **{'optn': False})
        bbo = s.BlackBox({'x':[x for x in np.linspace(-2,2,11)]})
        bbo = s.BlackBox([{'x':x} for x in np.linspace(-2,2,11)])
        bbo = s.BlackBox([ [x] for x in np.linspace(-2,2,11)])
        bbo = s.BlackBox([ [x] for x in np.linspace(-2,2,11)], True, optn=False)
        bbo = s.BlackBox([ [x] for x in np.linspace(-2,2,11)], optn1=True, optn2=False)


    # def test_edi_formulation_variable(self):
    #     "Tests the variable constructor in edi.formulation"



if __name__ == '__main__':
    unittest.main()
