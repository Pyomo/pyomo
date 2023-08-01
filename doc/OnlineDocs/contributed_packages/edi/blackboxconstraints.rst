Runtime (Black-Box) Constraints
===============================


Overview
--------

While some constraints are explicitly known and can be written directly into the optimization problem, it is common (particularly in engineering design) for some relationships to be too complex to be directly coded as a constraint.  

One of the main features of EDI is the streamlined implementation of Black-Box Constraints.  A *Black-Box* is defined as a routine that performs hidden computation not visible EDI, pyomo, or more generally the optimization algorithm.  However, it is **not** assumed that black-boxes are unable to return gradient information.  A black-box in this context may be capable of returning arbitrary derivative information.

Black-box constraints are considered to be a sub-class of a more general class of constraints called *Runtime Constraints*, that is constraints that are not actually constructed until the optimization routine is actually running.  In most cases, Runtime Constraints are approximated as linear by the solver, and therefore a Runtime Constraint is expected to provide function evaluations and gradient information.

The use of Runtime constraints requires a black box model that is discussed in detail in the dedicated documentation (see :doc:`here <./blackboxconstraints>`), but for the purposes of demonstrating the constructors, a simple black box example will appear in all of the code snippets below.

Runtime Constraints are declared one of two ways, just as regular constraints.  The ``f.RuntimeConstraint()`` constructor is available: 

.. code-block:: python

    import pyomo.environ as pyo
    from pyomo.environ import units
    from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel, BBVariable
    f = Formulation()
    x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'x variable')
    y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'y variable')
    z = f.Variable(name = 'z', guess = 1.0, units = 'm^2', description = 'Output var')
    f.Objective( x + y )
    class UnitCircle(BlackBoxFunctionModel):
        def __init__(self): 
            super(UnitCircle, self).__init__()
            self.description = 'This model evaluates the function: z = x**2 + y**2'
            self.inputs.append(BBVariable(  name = 'x', 
                                            size = 0, 
                                            units = 'ft' , 
                                            description = 'The x variable' ) )
            self.inputs.append(BBVariable(  name = 'y', 
                                            size = 0, 
                                            units = 'ft' , 
                                            description = 'The y variable' ) )
            self.outputs.append(BBVariable( name = 'z', 
                                            size = 0, 
                                            units = 'ft**2',  
                                            description = 'Output variable' ) )
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
    f.Constraint( z <= 1*units.m**2 )

    f.RuntimeConstraint( z, '==', [x,y], UnitCircle() )


The ``f.RuntimeConstraint()`` constructor takes in the following inputs:

.. py:function:: f.RuntimeConstraint(outputs, operators, inputs, black_box)

    Declares a runtime constraint in a pyomo.edi.formulation

    :param outputs: The outputs of the black box function
    :type outputs: pyomo.environ.Var or list or tuple
    :param operators: The operators that are used to construct constraints.  Currently, only equality constraints are supported and will be the default no matter what is passed in here (see `this issue <https://github.com/codykarcher/pyomo/issues/4>`__)
    :type operators: str or list or tuple
    :param inputs: The inputs to the black box function
    :type inputs: pyomo.environ.Var or list or tuple
    :param black_box: The object that stores the black-box function.   See the :doc:`black box constraint documentation <./blackboxconstraints>` for details on constructing this object
    :type black_box: pyomo.contrib.edi.BlackBoxFunctionModel


The following are alternative construction methods that may be of use:

.. code-block:: python

    f.RuntimeConstraint( *( z, '==', [x,y], UnitCircle() ) )


.. code-block:: python

    f.RuntimeConstraint( *[ z, '==', [x,y], UnitCircle() ] )

.. code-block:: python

    f.RuntimeConstraint( **{ 'outputs'   : z, 
                             'operators' : '==', 
                             'inputs'    : [x,y], 
                             'black_box' : UnitCircle() } )

.. code-block:: python

    f.RuntimeConstraint( *( [z], ['=='], [x,y], UnitCircle() ) )

However, more commonly we expect users to construct Runtime Constraints as a part of a ``f.ConstraintList()`` declaration.  Simply include a list, tuple, or dict as a part of the ConstraintList as follows:

.. code-block:: python

    import pyomo.environ as pyo
    from pyomo.environ import units
    from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel, BBVariable
    f = Formulation()
    x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'x variable')
    y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'y variable')
    z = f.Variable(name = 'z', guess = 1.0, units = 'm^2', description = 'Output var')
    f.Objective( x + y )
    class UnitCircle(BlackBoxFunctionModel):
        def __init__(self): 
            super(UnitCircle, self).__init__()
            self.description = 'This model evaluates the function: z = x**2 + y**2'
            self.inputs.append(BBVariable(  name = 'x', 
                                            size = 0, 
                                            units = 'ft' , 
                                            description = 'The x variable' ) )
            self.inputs.append(BBVariable(  name = 'y', 
                                            size = 0, 
                                            units = 'ft' , 
                                            description = 'The y variable' ) )
            self.outputs.append(BBVariable( name = 'z', 
                                            size = 0, 
                                            units = 'ft**2',  
                                            description = 'Output variable' ) )
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
            z <= 1*units.m**2 ) ,
            [ z, '==', [x,y], UnitCircle() ] ,             
        ]
    )

Any of the alternative declarations above are valid to pass into the ``f.ConstraintList()`` constructor, for example:

.. code-block:: python

    f.ConstraintList(
        [
            z <= 1*units.m**2 ) ,
            ( z, '==', [x,y], UnitCircle() ) ,             
        ]
    )

.. code-block:: python

    f.ConstraintList(
        [
            z <= 1*units.m**2 ) ,
            [ z, '==', [x,y], UnitCircle() ] ,             
        ]
    )

.. code-block:: python

    f.ConstraintList(
        [
            z <= 1*units.m**2 ) ,
            { 'outputs'   : z, 
              'operators' : '==', 
              'inputs'    : [x,y], 
              'black_box' : UnitCircle() } ,             
        ]
    )

.. code-block:: python

    f.ConstraintList(
        [
            z <= 1*units.m**2 ) ,
            ( [z], ['=='], [x,y], UnitCircle() ) ,             
        ]
    )


Known Issues
------------

* Currently only equality constraints are supported, pending an update to pyomo (see `this issue <https://github.com/codykarcher/pyomo/issues/2>`__)
* Runtime constraints must output to a variable, numbers and constants are not permitted (see `this issue <https://github.com/codykarcher/pyomo/issues/4>`__)
