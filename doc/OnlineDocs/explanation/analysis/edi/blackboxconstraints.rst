Runtime (Black-Box) Constraints
===============================


Overview
--------

While some constraints are explicitly known and can be written directly into the optimization problem, it is common (particularly in engineering design) for some relationships to be too complex to be directly coded as a constraint.  

EDI refers to these types of constraints as ``RuntimeConstraints`` because they are not constructed until they are needed by the solver.  A particular subset of Runtime Constraints of interest are Black-Box constraints, that is, constraints which call to an external routine.  To the average Pyomo and EDI user, ``RuntimeConstraints`` are (for all intents and purposes) Black-Box constraints, and the distinction is semantic.  

In other words, if you wish to code a black-box constraint using EDI, you will be using the Runtime Constraint constructor.

In this context, a *Black-Box* is defined as a routine that performs hidden computation not visible EDI, Pyomo, or more generally the optimization algorithm.  However, it is **not** assumed that black-boxes are unable to return gradient information.  A black-box in this context may be capable of returning arbitrary derivative information.


Construction
------------

Runtime constraints consist of two separate elements that need to be constructed: an ``__init__`` function and a ``BlackBox`` function.  Additionally, there are two pre-implemented functions that intermediate to advanced users will wish to interface with: ``BlackBox_Standardized`` and ``MultiCase``.


Constructing a Black Box
++++++++++++++++++++++++

First, we need to create an object which is visible to Pyomo/EDI that calls the black-box function.  EDI calls this a ``BlackBoxFunctionModel``, and it is a base class that gets inherited into the objects you will create as a user.

A simple example is shown below:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: RuntimeConstraints_Snippet_01
    :end-before: # END: RuntimeConstraints_Snippet_01

The inheriting classes can have any valid Python name (in this case ``Parabola``) and must have two methods ``__init__()`` and ``BlackBox()``.  


The init method
***************

The ``__init__()`` function sets up the model, and has 5 distinct steps.  First, the parent class ``__init__()`` must be called:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 16
    :start-after: # BEGIN: RuntimeConstraints_Snippet_02
    :end-before: # END: RuntimeConstraints_Snippet_02

In general, this line can be used verbatim.

Next, you must tell the model what its inputs are by appending them to the ``self.inputs`` attribute.  These inputs exist entirely in the local namespace of the black-box model, and are **independent** of the namespace in the optimization model (e.g. something called ``x`` in the optimization model can be called ``y`` in the black-box model).  Inputs must have a ``name`` and ``units`` and can optionally have a ``description``, and ``size``.

.. py:function:: self.inputs.append(name, units, description='', size=0)

    Appends a variable to a black box input list

   :param name: The name of the variable, any valid Python string.  **Does not** have to match the name in the optimization formulation
   :type  name: str
   :param units: The units of the variable.  Every entry in a vector variable must have the same units.  Entries of '', ' ', '-', 'None', and 'dimensionless' all become units.dimensionless. The units **must** be convertible to the units used in the optimization formulation (e.g., meters and feet), but **are not required** to be the same.  Because of this, for example, a black-box can be written in imperial units, while an optimization formulation operates in metric.
   :type  units: str or pyomo.core.base.units_container._PyomoUnit
   :param description: A description of the variable
   :type  description: str
   :param size: The size (or shape) of the variable.  Entries of 0, 1, and None all correspond to scalar variables.  Other integers correspond to vector variables.  Matrix and tensor variable are declared using lists of ints, ex: [10,10].  Matrix and tensor variables with a dimension of 1 (i.e., [10,10,1]) will be rejected as the extra dimension holds no meaningful value.  
   :type  size: int or list
   

Models with multiple inputs simply call the ``self.input.append()`` command multiple times:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 16
    :start-after: # BEGIN: RuntimeConstraints_Snippet_03
    :end-before: # END: RuntimeConstraints_Snippet_03

Input names must be unique, and an error is raised if a repeated name is attempted to be set.  

Next, outputs must be added to the model.  This is done identically to inputs, however the function is now ``self.outputs.append()``

.. py:function:: self.outputs.append(name, units, description='', size=0)

    Appends a variable to a black box output list

   :param name: The name of the variable, any valid Python string.  **Does not** have to match the name in the optimization formulation
   :type  name: str
   :param units: The units of the variable.  Every entry in a vector variable must have the same units.  Entries of '', ' ', '-', 'None', and 'dimensionless' all become units.dimensionless. The units **must** be convertible to the units used in the optimization formulation (ex, meters and feet), but **are not required** to be the same.  Because of this, for example, a black-box can be written in imperial units, while an optimization formulation operates in metric.
   :type  units: str or pyomo.core.base.units_container._PyomoUnit
   :param description: A description of the variable
   :type  description: str
   :param size: The size (or shape) of the variable.  Entries of 0, 1, and None all correspond to scalar variables.  Other integers correspond to vector variables.  Matrix and tensor variable are declared using lists of ints, ex: [10,10].  Matrix and tensor variables with a dimension of 1 (i.e., [10,10,1]) will be rejected as the extra dimension holds no meaningful value.  
   :type  size: int or list
   

and similarly:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 16
    :start-after: # BEGIN: RuntimeConstraints_Snippet_04
    :end-before: # END: RuntimeConstraints_Snippet_04


Finally, the highest available derivative must be set.  For models being used in optimization, this will most often be ``1``, i.e. first derivative, gradient, or Jacobian information.

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 16
    :start-after: # BEGIN: RuntimeConstraints_Snippet_05
    :end-before: # END: RuntimeConstraints_Snippet_05


The BlackBox method
*******************

The ``BlackBox`` is extremely flexible, but here we present standard usage for a typical user.

The ``BlackBox`` method should take in the inputs as arguments in the order defined during the ``__init__()`` method.  Note that the method assumes inputs are passed in **with units** and outputs are returned **with units**.  In general, the units on inputs and outputs need not be in any specific system, but should be convertible (e.g. meters and feet) to the units specified in the ``__init__()`` function. 

Various unpacking schemes are enabled by default via the ``parse_inputs`` function.  Use of this function is not necessary, but provides for the parsing of index argumented lists (ex: ``function(x1, x2, x3)``) and keyword argumented dictionaries (ex: ``function({'x2':x2, 'x1':x1, 'x3',x3})``), along with a few other possibilities.

The unit handling system in Pyomo can be rough at times, and so best practice is generally for the BlackBox function is expected to return values that are ``pint.Quantity`` types.  These are obtained using the ``pyo.as_quantity()`` function.

Since the units cannot be assumed on input, the first step in any black box is to convert to the model units:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 16
    :start-after: # BEGIN: RuntimeConstraints_Snippet_06
    :end-before: # END: RuntimeConstraints_Snippet_06


And frequently, it is a good idea to cast these to a float value using ``pyomo.environ.value``:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 16
    :start-after: # BEGIN: RuntimeConstraints_Snippet_07
    :end-before: # END: RuntimeConstraints_Snippet_07

The assumed units can now be added if desired, but this may cause a slowdown in performance.  Typical usage is to strip units then append at the end, unless many unit systems are being used in the actual computations.

Operations can now be performed to compute the output and derivatives as desired.

When preparing the outputs, note that all outputs must have units and be of type ``pint.Quantity``:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 16
    :start-after: # BEGIN: RuntimeConstraints_Snippet_08
    :end-before: # END: RuntimeConstraints_Snippet_08


There are multiple options for packing the output.  In general, the ``BlackBox`` method should output in a way that is convenient for the modeler.  For simple black boxes, with less than 10 scalar inputs and scalar outputs, it is probably easiest to output as a tuple, as was done in the example here.  Consider as a second example a function of form ``[u,v,w]=f(x,y,z)``.  The simple packing would be:

.. code-block:: python

    return [u,v,w], [[du_dx, du_dy, du_dz], [dv_dx, dv_dy, dv_dz], [dw_dx, dw_dy, dw_dz]]

If only one output, the following is also allowed:

.. code-block:: python

    return u, [du_dx, du_dy, du_dz]

For more complex models, it is encouraged to switch to the ``NamedTuple`` output also used by the ``BlackBox_Standardized`` method:

.. code-block:: python

    returnTuple = namedtuple('returnTuple', ['values', 'first', 'second'])
    optTuple    = namedtuple('optTuple', ['u','v','w'])
    iptTuple    = namedtuple('iptTuple', ['x','y','z'])

    values = optTuple(u,v,w)
    first  = optTuple( iptTuple(du_dx, du_dy, du_dz),
                       iptTuple(dv_dx, dv_dy, dv_dz), 
                       iptTuple(dw_dx, dw_dy, dw_dz) )
    second = None # Second derivatives not currently supported

    return returnTuple(values,first,second)

Dictionaries with the same keywords are also supported:

.. code-block:: python

    values = {'u':u , 'v':v , 'w':w}
    first  = { 'u': {'x':du_dx ,'y':du_dy 'z':du_dz},
               'v': {'x':dv_dx ,'y':dv_dy 'z':dv_dz}, 
               'w': {'x':dw_dx ,'y':dw_dy 'z':dw_dz} }
    second = None # Second derivatives not currently supported

    return { 'values': values,
             'first':  first,
             'second': second }

As are combinations of any of these options.

In the event that the inputs and/or outputs are non-scalar, then outputs should be passed out as structures of their appropriate shape.  Derivatives are a little more complicated.  If the input **or** output is a scalar, then the derivative ``du_dx`` should have the shape of the non-scalar input/output.  However, if **both** are non-scalar, then the output should be a numpy array that takes in indices of the inputs as indices.  For example, an input (x) of dimension 2x2 and an output (u) of dimension 4x4x4, then the derivative information would be packed as:

.. code-block:: python

    du_dx[0,0,0,0,0] # derivative of u[0,0,0] with respect to x[0,0]
    du_dx[0,0,1,0,0] # derivative of u[0,0,1] with respect to x[0,0]
    du_dx[0,0,0,1,1] # derivative of u[0,0,0] with respect to x[1,1]

Note that this may change in the future, as developers are currently unsatisfied with extensions of this method to second order and higher derivatives.

The BlackBox_Standardized Method
********************************
The ``BlackBox`` method is designed to be highly flexible for use by practicing engineers.  However, integrating these black box analysis tools into optimization often requires a common, structured framework for the output.  The ``BlackBox_Standardized`` method provides this common interface.  

The ``BlackBox_Standardized`` method will **always** provide an output that is a nested series of ``NamedTuples``:

- [0]--'values'
    - [0]--'name_of_first_output': First output of the black box
    - [1]--'name_of_first_output': Second output of the black box
    - ...
    - [n]--'name_of_last_output': Last output of the black box
- [1]--'first'
    - [0]--'name_of_first_output'
        - [0]--'name_of_first_input': Derivative of the first output of the black box wrt. the first input of the black box
        - ...
        - [n]--'name_of_last_input': Derivative of the first output of the black box wrt. the last input of the black box
    - ...
    - [n]--'name_of_last_output'
        - [0]--'name_of_first_input': Derivative of the last output of the black box wrt. the first input of the black box
        - ...
        - [n]--'name_of_last_input': Derivative of the last output of the black box wrt. the last input of the black box
- [2]--'second'
    - At present, this will always be ``None``, though second order support is planned in future versions

For example, for a black box function ``[u,v,w] = f(x,y,z)``, the ``BlackBox_Standardized`` method would look as follows:

.. code-block:: python

    opt = m.BlackBox_Standardized(x,y,z)
    opt.values.u    # The output u
    opt[0][2]       # The output w
    opt.first.u.x   # The derivative du/dx
    opt.first.w.y   # The derivative dw/dy 
    opt[1][1][2]    # The derivative dv/dz
    opt.second      # Will be None in the current version
    opt[2]          # Will be None in the current version

Note that while the current implementation does have a default ``BlackBox_Standardized`` method, this method is written for robustness and not performance.  For users who want maximum performance, we recommend writing your own ``BlackBox_Standardized`` method and overriding the base class.  Performance improvements will be most significant when a model has a large number of non-scalar inputs and/or outputs.

Note also that the names in the named tuple are the names used in the black box model and **not** the names in the optimization problem.  These two namespaces are intentionally separated.  If you wish to use the namespace access, you must either get the names from ``[x.name for x in m.inputs]`` and ``[x.name for x in m.outputs]`` and use python's ``getattr`` function, or else have some other prior knowledge of the local black box namespace.

Below is an example using the default ``BlackBox_Standardized`` method:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: RuntimeConstraints_Snippet_11
    :end-before: # END: RuntimeConstraints_Snippet_11

And now if we wish to implement our own custom ``BlackBox_Standardized`` method:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: RuntimeConstraints_Snippet_12
    :end-before: # END: RuntimeConstraints_Snippet_12

Note that if you are writing solvers for EDI, you should **exclusively** be calling the ``BlackBox_Standardized`` method, and never the basic ``BlackBox`` method, as this will ensure you are always working with a predictable data structure.

As a modeler, you should **always** verify the output of ``BlackBox_Standardized`` as a final check in your effort to ensure it is packing the data as intended.  Please notify the developers of any issues.

The MultiCase Method
********************
The ``MultiCase`` method provides a native capability to call the ``BlackBox`` method across multiple inputs simultaneously.  This function is **not** vectorized in the base class and is **not** optimized for performance.  If you wish to have a high performance vectorized function, you will need to implement your own method.

Inputs to the ``MultiCase`` function should be a list of cases, which can be packed in any form accepted by the ``BlackBox_Standardized`` method.  Overloading these functions may allow different forms of unpacking scheme.

The output is a list of ``NamedTuple`` objects that are output from the ``BlackBox_Standardized`` method.  If overloading, you may choose to output via a different packing scheme.

Below is an example of overriding the default ``MultiCase`` method:


.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: RuntimeConstraints_Snippet_13
    :end-before: # END: RuntimeConstraints_Snippet_13


Including a Black-Box in an EDI Formulation
+++++++++++++++++++++++++++++++++++++++++++

This second construction step is covered in the :doc:`Formulation <./formulation>` documentation, but is repeated here for completion.  Future versions may differentiate this section.

Runtime Constraints are declared one of two ways, just as regular constraints.  The ``f.RuntimeConstraint()`` constructor is available: 

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_09
    :end-before: # END: Formulation_Snippet_09


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

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_10
    :end-before: # END: Formulation_Snippet_10

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_11
    :end-before: # END: Formulation_Snippet_11

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_12
    :end-before: # END: Formulation_Snippet_12

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_13
    :end-before: # END: Formulation_Snippet_13

However, more commonly we expect users to construct Runtime Constraints as a part of a ``f.ConstraintList()`` declaration.  Simply include a list, tuple, or dict as a part of the ConstraintList as follows:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_14
    :end-before: # END: Formulation_Snippet_14

Any of the alternative declarations above are valid to pass into the ``f.ConstraintList()`` constructor, for example:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_15
    :end-before: # END: Formulation_Snippet_15

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_16
    :end-before: # END: Formulation_Snippet_16

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_17
    :end-before: # END: Formulation_Snippet_17

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_18
    :end-before: # END: Formulation_Snippet_18



Examples
--------

More examples will be added over time.  Feel free to reach out to the developers if you have questions regarding model development.

A standard construction
+++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: RuntimeConstraints_Snippet_10
    :end-before: # END: RuntimeConstraints_Snippet_10


Tips
----

* Use the Pyomo ``tostr()`` function (``from pyomo.common.formatting import tostr``) to print the results of black-boxes for more meaningful printouts
* Align input and output declarations just as is recommended for optimization variable and constant declarations
* Declare an input/output all on one line, no matter what the style guides say
* * This interface is designed for subject matter experts who are not Python users to have a simple, easy path to include their tools/models into a Python based optimization architecture.  
* Embrace units.  They will save you so many times, it is well worth the minor additional overhead
* Pyomo units work slightly differently than pint (for those with pint experience), but those differences should be hidden from the model creator for the most part
* It is common to use this framework to call to a piece of software external to Python
* A model summary can be printed by calling ``print(model_instance.summary)``


Known Issues
------------

* Currently only equality constraints are supported, pending an update to Pyomo (see `this issue <https://github.com/codykarcher/pyomo/issues/2>`__)
* Runtime constraints must output to a variable, numbers and constants are not permitted (see `this issue <https://github.com/codykarcher/pyomo/issues/4>`__)
* This functionality is not well tested when returning derivatives higher than first order.  Though it should work, exercise caution and reach out to the dev team if questions arise.
