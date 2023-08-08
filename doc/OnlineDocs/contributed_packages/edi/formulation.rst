Formulation
===========

.. |br| raw:: html

   <br />


The core object in EDI is called a *Formulation*.  For experienced pyomo users, a Formulation inherits from a pyomo *ConcreteModel*, and can therefore be treated exactly as a typical *ConcreteModel* with a few additional features.  

Each modeling element (ex: Variable, Constant, Objective, and Constraint) has a constructor that is used to create the corresponding element in the *Formulation* instance.  In addition, there are a number of helper functions that collect and return model elements or perform supporting actions.

Construction
------------
A *Formulation* is constructed as follows:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_01
    :end-before: # END: Formulation_Snippet_01


Standard practice is to construct a formulation to namespace variable ``f``, but any valid python name can be used.  Standard pyomo practice would be to construct this to ``model`` or ``m``.


Declaring Variables
-------------------

See the :doc:`Variables <./variables>` Documentation

Variables are declared using the ``f.Variable()`` function.  This function creates an instance of ``pyomo.environ.Var`` and adds it to the ``edi.contrib.Formulation``.  The function returns an instance of a ``pyomo.environ.Var`` that can be used in later construction.

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_02
    :end-before: # END: Formulation_Snippet_02


Declaring Constants
-------------------

See the :doc:`Constants <./constants>` Documentation

Constants (referred to in base pyomo as parameters or ``Params``) are declared using the ``f.Constant()`` function.  This function creates an instance of a ``pyomo.environ.Param`` and adds it to the ``edi.contrib.Formulation``.  This function also returns an instance of ``pyomo.environ.Param`` that can be used in later construction.

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_03
    :end-before: # END: Formulation_Snippet_03



Declaring Objectives
--------------------

See the :doc:`Objectives <./objectives>` Documentation

Objectives are declared using the ``f.Objective()`` function.  This function creates an instance of a ``pyomo.environ.Objective`` and adds it to the ``edi.contrib.Formulation``.  Multiple objectives can be declared, but interpretation of multiple objectives will depend on the solver.  The returned values of the ``f.Variable()`` and ``f.Constant()`` declarations can be used to construct the objective.  Black-box (ie, Runtime) objectives are not supported at this time, but are planned in a future update.

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_04
    :end-before: # END: Formulation_Snippet_04


By default, objectives are minimized, but can be switched to a maximize using the ``sense`` keyword from pyomo:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_05
    :end-before: # END: Formulation_Snippet_05


Note:  Future version will allow a string to be passed into ``sense`` (see `this issue <https://github.com/codykarcher/pyomo/issues/7>`_)


Declaring Constraints
---------------------

See the :doc:`Constraints <./constraints>` Documentation

Constraints can be declared in two ways.  First is using the standard ``f.Constraint()`` constructor.  This function creates an instance of ``pyomo.environ.Constraint`` and adds it to the ``edi.contrib.Formulation``.  The operators ``<=``, ``>=``, and ``==`` are used as constraint constructors.

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_06
    :end-before: # END: Formulation_Snippet_06

Constraints can also be declared using the ``f.ConstraintList()`` function.  This function takes in a list of constraints and allows for multiple constraints to be declared in one go:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_07
    :end-before: # END: Formulation_Snippet_07

The constraint list can also be declared a priori and passed in to the ``f.ConstraintList()`` function, which may be beneficial for complex models:

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_08
    :end-before: # END: Formulation_Snippet_08

We recommend that most users should be using the ``f.ConstraintList()`` function, with the ``f.Constraint()`` function being reserved for under-the-hood usage.


Declaring Runtime (Black-Box) Constraints
-----------------------------------------

See the :doc:`Runtime (Black-Box) Constraints <./blackboxconstraints>` Documentation

One of the main features of EDI is the streamlined implementation of Black-Box Constraints.  A *Black-Box* is defined as a routine that performs hidden computation not visible EDI, pyomo, or more generally the optimization algorithm.  However, it is **not** assumed that black-boxes are unable to return gradient information.  A black-box in this context may be capable of returning arbitrary derivative information.

Black-box constraints are considered to be a sub-class of a more general class of constraints called *Runtime Constraints*, that is constraints that are not actually constructed until the optimization routine is actually running.  In most cases, Runtime Constraints are approximated as linear by the solver, and therefore a Runtime Constraint is expected to provide function evaluations and gradient information.

The use of Runtime constraints requires a black box model that is discussed in detail in the dedicated documentation (see :doc:`here <./blackboxconstraints>`), but for the purposes of demonstrating the constructors, a simple black box example will appear in all of the code snippets below.

Runtime Constraints are declared one of two ways, just as regular constraints.  The ``f.RuntimeConstraint()`` constructor is available: 

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Formulation_Snippet_09
    :end-before: # END: Formulation_Snippet_09


The ``f.RuntimeConstraint()`` constructor takes in the following inputs:

.. py:function:: f.RuntimeConstraint(outputs, operators, inputs, black_box)
    :noindex:

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

Support Functions
-----------------

``f.get_variables()``  |br|
Returns a list variables that have been defined in the formulation in declaration order.  Will only return variables defined via EDI.

``f.get_constants()``  |br|
Returns a list of constants that have been defined in the formulation in declaration order.  Will only return constants/parameters defined via EDI.

``f.get_objectives()``  |br|
Returns a list of objectives that have been defined in the formulation in declaration order.  Will only return objectives defined via EDI.

``f.get_constraints()``  |br|
Returns a list of constraints that have been defined in the formulation in declaration order.  This command returns a list that includes both explicit and runtime (black-box) constraints, but only constraints that have been defined via EDI.

``f.get_explicitConstraints()``  |br|
Returns a list of *explicit* constraints that have been defined in the formulation in declaration order.  This command returns a list that includes *only* the explicit constraints and *not* the runtime (black-box) constraints.  Only includes constraints that have been defined via EDI.

``f.get_runtimeConstraints()``  |br|
Returns a list of *runtime* (ie. black-box) constraints that have been defined in the formulation in declaration order.  This command returns a list that includes *only* the runtime constraints and *not* the explicit constraints.  Only includes constraints that have been defined via EDI.

``f.check_units()``  |br|
Checks the units of each objective and constraint for consistency.  Will only check objectives and constraints defined via EDI.
