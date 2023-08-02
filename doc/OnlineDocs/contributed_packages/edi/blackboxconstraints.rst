Runtime (Black-Box) Constraints
===============================


Overview
--------

While some constraints are explicitly known and can be written directly into the optimization problem, it is common (particularly in engineering design) for some relationships to be too complex to be directly coded as a constraint.  

EDI refers to these types of constraints as ``RuntimeConstraints`` because they are not constructed until they are needed by the solver.  A particular subset of Runtime Constraints of interest are Black-Box constraints, that is, constraints which call to an external routine.  To the average pyomo and EDI user, ``RuntimeConstraints`` are for all intents and purposes Black-Box constraint, and the distinction is semantic.  

In other words, if you wish to code a black-box constraint using EDI, you will be using the Runtime Constraint constructor.

In this context, a *Black-Box* is defined as a routine that performs hidden computation not visible EDI, pyomo, or more generally the optimization algorithm.  However, it is **not** assumed that black-boxes are unable to return gradient information.  A black-box in this context may be capable of returning arbitrary derivative information.


Construction
------------

Runtime constraints consist of two separate elements that need to be constructed.  


Constructing a Black Box
++++++++++++++++++++++++

First, we need to create an object which is visible to pyomo/EDI that calls the black-box function.  EDI calls this a ``BlackBoxFunctionModel``, and it is a base class that gets inherited into the objects you will create as a user.

A simple example is shown below:

.. code-block:: python

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
                self.inputs.append(
                    name='x', units='ft' , description='The x variable'
                )

                # Append the model outputs
                self.outputs.append(
                    name='y', units='ft**2' , description='The y variable'
                )
                
                # Set the highest available derivative
                # Should be 1 for most cases but defauts to 0
                self.availableDerivative = 1

            def BlackBox(self, x): # The actual function that does things

                # Convert to correct units and cast to a float
                x = pyo.value(units.convert(x,self.inputs['x'].units))

                # Compute y
                y = x**2

                # Compute dy/dx
                dydx = 2*x
                
                # Add the units to the output
                y = y * self.inputs['y'].units
                
                # Add the units to the derivative for output
                dydx = dydx * self.inputs['y'].units / self.inputs['x'].units
                
                # Return
                return y, [dydx] # return z, grad(z), hess(z)...


The inheriting classes can have any valid python name (in this case ``Parabola``) and have two methods ``__init__()`` and ``BlackBox()``.  


The init method
***************

The ``__init__()`` function sets up the model, and has 5 distinct steps.  First, the parent class ``__init__()`` must be called:

.. code-block:: python
                
    super().__init__()

In general, this line can be used verbatim.

Next, you must tell the model what its inputs are by appending them to the ``self.inputs`` attribute.  These inputs exist entirely in the local namespace of the black-box model, and are **independent** of the namespace in the optimization model (ex, something called ``x`` in the optimization can be called ``y`` in the black-box model).  Inputs must have a ``name`` and ``units``, and has optional arguments ``description``, and ``size``, all of which are defined the same way as EDI variables.  (There are some advanced uses reserved for the advanced discussion).

.. py:function:: self.inputs.append(name, units, description='', size=0)

    Appends a variable to a black box input list

   :param name: The name of the variable, any valid python string.  **Does not** have to match the name in the optimization formulation
   :type  name: str
   :param units: The units of the variable.  Every entry in a vector variable must have the same units.  Entries of '', ' ', '-', 'None', and 'dimensionless' all become units.dimensionless. The units **must** be convertable to the units used in the optimization formulation (ex, meters and feet), but **are not required** to be the same.  Because of this, for example, a black-box can be written in imperial units, while an optimization formulation operates in metric.
   :type  units: str or pyomo.core.base.units_container._PyomoUnit
   :param description: A description of the variable
   :type  description: str
   :param size: The size (or shape) of the variable.  Entries of 0, 1, and None all correspond to scalar variables.  Other integers correspond to vector variables.  Matrix and tensor variable are declared using lists of ints, ex: [10,10].  Matrix and tensor variables with a dimension of 1 (ie, [10,10,1]) will be rejected as the extra dimension holds no meaningful value.  
   :type  size: int or list
   

Models with multiple inputs simply call the ``self.input.append()`` command multiple times:

.. code-block:: python

    # Append the model inputs
    self.inputs.append( name='x', units='ft' , description='The x variable' )
    self.inputs.append( name='y', units='ft' , description='The y variable' )

Input names must be unique, and an error is raised if a repeated name is attempted to be set.  


Next, outputs must be added to the model.  This is done identically to inputs, however the function is now ``self.outputs.append()``

.. py:function:: self.outputs.append(name, units, description='', size=0)

    Appends a variable to a black box output list

   :param name: The name of the variable, any valid python string.  **Does not** have to match the name in the optimization formulation
   :type  name: str
   :param units: The units of the variable.  Every entry in a vector variable must have the same units.  Entries of '', ' ', '-', 'None', and 'dimensionless' all become units.dimensionless. The units **must** be convertable to the units used in the optimization formulation (ex, meters and feet), but **are not required** to be the same.  Because of this, for example, a black-box can be written in imperial units, while an optimization formulation operates in metric.
   :type  units: str or pyomo.core.base.units_container._PyomoUnit
   :param description: A description of the variable
   :type  description: str
   :param size: The size (or shape) of the variable.  Entries of 0, 1, and None all correspond to scalar variables.  Other integers correspond to vector variables.  Matrix and tensor variable are declared using lists of ints, ex: [10,10].  Matrix and tensor variables with a dimension of 1 (ie, [10,10,1]) will be rejected as the extra dimension holds no meaningful value.  
   :type  size: int or list
   

and similarly:

.. code-block:: python

    # Append the model inputs
    self.outputs.append( name='u', units='ft' , description='The u variable' )
    self.outputs.append( name='v', units='ft' , description='The v variable' )


Finally, the highest available derivative must be set.  For models being used in optimization, this will most often be ``1``, ie first derivative, gradient, or Jacobian information.


.. code-block:: python

    # Set the highest available derivative
    # Should be 1 for most cases but defauts to 0
    self.availableDerivative = 1


The BlackBox method
*******************

The ``BlackBox`` is extremely flexible, but here we present standard usage for a beginner user. Advanced users should also check the :doc:`advanced <./advancedruntimeconstraints>` documentation for more flexible ways of building black-box models

The ``BlackBox`` method assumes to take in the inputs as arguments in the order defined during the ``__init__()`` method.  Note that the method assumes inputs **with units** and expects outputs **with units**.  In general, the units on inputs and outputs need not be in any specific system, but should be convertible (ex, meters and feet) to whatever has been specified as the input units when defining in the ``__init__()`` function.  

Since the units cannot be assumed on input, the first step in any black box is to convert to the model units:

.. code-block:: python

    from pyomo.environ import units
    x = units.convert(x,self.inputs['x'].units)


And frequently, it is a good idea to cast these to a float value using ``pyomo.environ.value``:

.. code-block:: python
        
        import pyomo.environ as pyo
        from pyomo.environ import units
        x = pyo.value(units.convert(x,self.inputs['x'].units))

The assumed units can now be added if desired, but this may cause a slowdown in performance.

Operations can now be performed to compute the output and derivatives as desired.

When preparing the outputs, note that all outputs must have units:

.. code-block:: python

    # Add the units to the output
    y = y * self.inputs['y'].units
    
    # Add the units to the derivative for output
    dydx = dydx * self.inputs['y'].units / self.inputs['x'].units


The ``BlackBox`` method then outputs a tuple of length ``self.availableDerivative+1``.  Entry [0] is the values specified during the ``__init__()``, entry [1] is first derivative information, and similar for higher order if available.

.. code-block:: python
    
    # Return
    return y, [dydx] # return z, grad(z), hess(z)...


The full unpacking is as follows:

::

    output[0] = <list_of_outputs>
    output[0][<index_of_output>] = <output>

    output[1] = <list_of_jacobians>
    output[1][<index_of_output>][<index_of_input>] = <d(output_of_specified_index)/d(input_of_specified_index)>


if there is one single scalar output, the unpacking may be simplified to the following (as shown in the above example)

::

    output[0] = <output>
    output[0][<index_of_output>] = <output>

    output[1] = <list_of_jacobians>
    output[1][<index_of_input>] = <d(output)/d(input_of_specified_index)>


See the :doc:`advanced <./advancedruntimeconstraints>` documentation for cases where the inputs and outputs are not scalar.


Including a Black-Box in an EDI Formulation
+++++++++++++++++++++++++++++++++++++++++++

This second construction step is covered in the :doc:`Formulation <./formulation>` documentation, but is repeated here for completion.

Runtime Constraints are declared one of two ways, just as regular constraints.  The ``f.RuntimeConstraint()`` constructor is available: 

.. code-block:: python

    import pyomo.environ as pyo
    from pyomo.environ import units
    from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel
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
            self.outputs.append name = 'z', 
                                units = 'ft**2',  
                                description = 'Output variable' ) 
            self.availableDerivative = 1
            self.post_init_setup(len(self.inputs))
        def BlackBox(self, x, y): # The actual function that does things
            # Converts to correct units then casts to float
            x = pyo.value(units.convert(x,self.inputs['x'].units))
            y = pyo.value(units.convert(y,self.inputs['y'].units))
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
    from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel
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
            self.outputs.append name = 'z', 
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

Examples
--------

A standard declaration statement
++++++++++++++++++++++++++++++++

.. code-block:: python

    from pyomo.environ import units
    from pyomo.contrib.edi import Formulation
    f = Formulation()
    x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
    y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')
    c = f.Constant(name = 'c', value = 1.0, units = ''   , description = 'A constant c' )
    f.Objective( c*x + y )
    f.ConstraintList(
        [
            x**2 + y**2 <= 1.0*units.m**2 ,
            x <= 0.75*units.m,
            x >= y ,
        ]
    )


Tips
----

* 


Known Issues
------------

* Currently only equality constraints are supported, pending an update to pyomo (see `this issue <https://github.com/codykarcher/pyomo/issues/2>`__)
* Runtime constraints must output to a variable, numbers and constants are not permitted (see `this issue <https://github.com/codykarcher/pyomo/issues/4>`__)
