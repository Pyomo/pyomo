Manipulating Pyomo Models
=========================

This section gives an overview of commonly used scripting commands when
working with Pyomo models. These commands must be applied to a concrete
model instance or in other words an instantiated model.


Repeated Solves
---------------

.. doctest::

   >>> import pyomo.environ as pyo
   >>> from pyomo.opt import SolverFactory
   >>> model = pyo.ConcreteModel()
   >>> model.nVars = pyo.Param(initialize=4)
   >>> model.N = pyo.RangeSet(model.nVars)
   >>> model.x = pyo.Var(model.N, within=pyo.Binary)
   >>> model.obj = pyo.Objective(expr=pyo.summation(model.x))
   >>> model.cuts = pyo.ConstraintList()
   >>> opt = SolverFactory('glpk')
   >>> opt.solve(model) # doctest: +SKIP

   >>> # Iterate, adding a cut to exclude the previously found solution
   >>> for i in range(5):
   ...    expr = 0
   ...    for j in model.x:
   ...        if pyo.value(model.x[j]) < 0.5:
   ...            expr += model.x[j]
   ...        else:
   ...            expr += (1 - model.x[j])
   ...    model.cuts.add( expr >= 1 )
   ...    results = opt.solve(model)
   ...    print ("\n===== iteration",i)
   ...    model.display() # doctest: +SKIP

To illustrate Python scripts for Pyomo we consider an example that is in
the file ``iterative1.py`` and is executed using the command

::

  python iterative1.py

.. note::

   This is a Python script that contains elements of Pyomo, so it is
   executed using the ``python`` command.  The ``pyomo`` command can be
   used, but then there will be some strange messages at the end when
   Pyomo finishes the script and attempts to send the results to a
   solver, which is what the ``pyomo`` command does.

This script creates a model, solves it, and then adds a constraint to
preclude the solution just found. This process is repeated, so the
script finds and prints multiple solutions.  The particular model it
creates is just the sum of four binary variables. One does not need a
computer to solve the problem or even to iterate over solutions. This
example is provided just to illustrate some elementary aspects of
scripting.

.. literalinclude:: /src/scripting/iterative1.spy
   :language: python

Let us now analyze this script. The first line is a comment that happens
to give the name of the file. This is followed by two lines that import
symbols for Pyomo. The pyomo namespace is imported as
``pyo``. Therefore, ``pyo.`` must precede each use of a Pyomo name.

.. literalinclude:: /src/scripting/iterative1_Import_symbols_for_pyomo.spy
   :language: python

An object to perform optimization is created by calling
``SolverFactory`` with an argument giving the name of the solver. The
argument would be ``'gurobi'`` if, e.g., Gurobi was desired instead of
glpk:

.. literalinclude:: /src/scripting/iterative1_Call_SolverFactory_with_argument.spy
   :language: python

The next lines after a comment create a model. For our discussion here,
we will refer to this as the base model because it will be extended by
adding constraints later. (The words "base model" are not reserved
words, they are just being introduced for the discussion of this
example).  There are no constraints in the base model, but that is just
to keep it simple.  Constraints could be present in the base model.
Even though it is an abstract model, the base model is fully specified
by these commands because it requires no external data:

.. literalinclude:: /src/scripting/iterative1_Create_base_model.spy
   :language: python

The next line is not part of the base model specification. It creates an
empty constraint list that the script will use to add constraints.

.. literalinclude:: /src/scripting/iterative1_Create_empty_constraint_list.spy
   :language: python

The next non-comment line creates the instantiated model and refers to
the instance object with a Python variable ``instance``.  Models run
using the ``pyomo`` script do not typically contain this line because
model instantiation is done by the ``pyomo`` script. In this example,
the ``create`` function is called without arguments because none are
needed; however, the name of a file with data commands is given as an
argument in many scripts.

.. literalinclude:: /src/scripting/iterative1_Create_instantiated_model.spy
   :language: python

The next line invokes the solver and refers to the object contain
results with the Python variable ``results``.

.. literalinclude:: /src/scripting/iterative1_Solve_and_refer_to_results.spy
   :language: python

The solve function loads the results into the instance, so the next line
writes out the updated values.

.. literalinclude:: /src/scripting/iterative1_Display_updated_value.spy
   :language: python

The next non-comment line is a Python iteration command that will
successively assign the integers from 0 to 4 to the Python variable
``i``, although that variable is not used in script. This loop is what
causes the script to generate five more solutions:

.. literalinclude:: /src/scripting/iterative1_Assign_integers.spy
   :language: python

An expression is built up in the Python variable named ``expr``.  The
Python variable ``j`` will be iteratively assigned all of the indexes of
the variable ``x``. For each index, the value of the variable (which was
loaded by the ``load`` method just described) is tested to see if it is
zero and the expression in ``expr`` is augmented accordingly.  Although
``expr`` is initialized to 0 (an integer), its type will change to be a
Pyomo expression when it is assigned expressions involving Pyomo
variable objects:

.. literalinclude:: /src/scripting/iterative1_Iteratively_assign_and_test.spy
   :language: python

During the first iteration (when ``i`` is 0), we know that all values of
``x`` will be 0, so we can anticipate what the expression will look
like. We know that ``x`` is indexed by the integers from 1 to 4 so we
know that ``j`` will take on the values from 1 to 4 and we also know
that all value of ``x`` will be zero for all indexes so we know that the
value of ``expr`` will be something like

::

  0 + instance.x[1] + instance.x[2] + instance.x[3] + instance.x[4]

The value of ``j`` will be evaluated because it is a Python variable;
however, because it is a Pyomo variable, the value of ``instance.x[j]``
not be used, instead the variable object will appear in the
expression. That is exactly what we want in this case. When we wanted to
use the current value in the ``if`` statement, we used the ``value``
function to get it.

The next line adds to the constraint list called ``c`` the requirement
that the expression be greater than or equal to one:

.. literalinclude:: /src/scripting/iterative1_Add_expression_constraint.spy
   :language: python

The proof that this precludes the last solution is left as an exercise
for the reader.

The final lines in the outer for loop find a solution and display it:

.. literalinclude:: /src/scripting/iterative1_Find_and_display_solution.spy
   :language: python

.. note::
   
   The assignment of the solve output to a results object is somewhat
   anachronistic. Many scripts just use

   >>> opt.solve(instance) # doctest: +SKIP

   since the results are moved to the instance by default, leaving
   the results object with little of interest. If, for some reason,
   you want the results to stay in the results object and *not* be
   moved to the instance, you would use

   >>> results = opt.solve(instance, load_solutions=False) # doctest: +SKIP
   
   This approach can be useful if there is a concern that the solver
   did not terminate with an optimal solution. For example,
   
   >>> results = opt.solve(instance, load_solutions=False) # doctest: +SKIP
   >>> if results.solver.termination_condition == TerminationCondition.optimal: # doctest: +SKIP
   ...     instance.solutions.load_from(results) # doctest: +SKIP

Changing the Model or Data and Re-solving
-----------------------------------------

The ``iterative1.py`` example above illustrates how a model can be changed and
then re-solved. In that example, the model is changed by adding a
constraint, but the model could also be changed by altering the values
of parameters. Note, however, that in these examples, we make the
changes to the concrete model instances.  This is particularly important
for ``AbstractModel`` users, as this implies working with the
``instance`` object rather than the ``model`` object, which allows us to
avoid creating a new ``model`` object for each solve. Here is the basic
idea for users of an ``AbstractModel``:

#. Create an ``AbstractModel`` (suppose it is called ``model``)
#. Call ``model.create_instance()`` to create an instance (suppose it is called ``instance``)
#. Solve ``instance``
#. Change something in ``instance``
#. Solve ``instance`` again

.. note::

   Users of ``ConcreteModel`` typically name their models ``model``, which
   can cause confusion to novice readers of documentation. Examples based on
   an ``AbstractModel`` will refer to ``instance`` where users of a
   ``ConcreteModel`` would typically use the name ``model``.

..
   NOTE: the tests in this file are fragile right now because some
   code has been brought inline (particularly from spy4scripts.spy)
   DO NOT redefine instance until you are ready to delete this entire comment.
   (and note that model is redefined all over the place).
   
.. testsetup:: *

    # code from spy4scripts
    import pyomo.environ as pyo

    instance = pyo.ConcreteModel()
    instance.I = pyo.Set(initialize=[1,2,3])
    instance.sigma = pyo.Param(mutable=True, initialize=2.3)
    instance.Theta = pyo.Param(instance.I, mutable=True)
    for i in instance.I:
        instance.Theta[i] = i
    idx = 1
    NewVal = 1134
    
    instance.x = pyo.Var([1,2,3], initialize=0)
    instance.y = pyo.Var()
    instance.iVar = pyo.Var([1,2,3], initialize=1, domain=pyo.Boolean)
    instance.sVar = pyo.Var(initialize=1, domain=pyo.Boolean)

If ``instance`` has a parameter whose name is ``Theta`` that was
declared to be ``mutable`` (i.e., ``mutable=True``) with an
index that contains ``idx``, then the value in ``NewVal`` can be assigned to
it using

   >>> instance.Theta[idx] = NewVal	       

For a singleton parameter named ``sigma`` (i.e., if it is not
indexed), the assignment can be made using

   >>> instance.sigma = NewVal

.. note::

   If the ``Param`` is not declared to be mutable, an error will occur if an assignment to it is attempted.
    
For more information about access to Pyomo parameters, see the section
in this document on ``Param`` access :ref:`ParamAccess`. Note that for
concrete models, the model is the instance.

Fixing Variables and Re-solving
-------------------------------

Instead of changing model data, scripts are often used to fix variable
values. The following example illustrates this.

.. literalinclude:: /src/scripting/iterative2.spy
   :language: python

In this example, the variables are binary. The model is solved and then
the value of ``model.x[2]`` is flipped to the opposite value before
solving the model again. The main lines of interest are:

.. literalinclude:: /src/scripting/iterative2_Flip_value_before_solve_again.spy
   :language: python

This could also have been accomplished by setting the upper and lower
bounds:

   >>> if instance.x[2].value == 0:
   ...     instance.x[2].setlb(1)
   ...     instance.x[2].setub(1)
   ... else:
   ...     instance.x[2].setlb(0)
   ...     instance.x[2].setub(0)
   
Notice that when using the bounds, we do not set ``fixed`` to ``True``
because that would fix the variable at whatever value it presently has
and then the bounds would be ignored by the solver.

For more information about access to Pyomo variables, see the section in
this document on ``Var`` access :ref:`VarAccess`.

Note that

   >>> instance.x.fix(1)

is equivalent to

   >>> instance.x.value = 1
   >>> instance.x.fixed = True

and
   >>> instance.x.fix()

is equivalent to

   >>> instance.x.fixed = True

Extending the Objective Function
--------------------------------

One can add terms to an objective function of a ``ConcreteModel`` (or
and instantiated ``AbstractModel``) using the ``expr`` attribute
of the objective function object. Here is a simple example:

.. doctest::

   >>> import pyomo.environ as pyo
   >>> from pyomo.opt import SolverFactory

   >>> model = pyo.ConcreteModel()

   >>> model.x = pyo.Var(within=pyo.PositiveReals)
   >>> model.y = pyo.Var(within=pyo.PositiveReals)

   >>> model.sillybound = pyo.Constraint(expr = model.x + model.y <= 2)

   >>> model.obj = pyo.Objective(expr = 20 * model.x)

   >>> opt = SolverFactory('glpk') # doctest: +SKIP
   >>> opt.solve(model) # doctest: +SKIP

   >>> model.pprint() # doctest: +SKIP

   >>> print ("------------- extend obj --------------") # doctest: +SKIP
   >>> model.obj.expr += 10 * model.y

   >>> opt.solve(model) # doctest: +SKIP
   >>> model.pprint() # doctest: +SKIP

Activating and Deactivating Objectives
--------------------------------------

Multiple objectives can be declared, but only one can be active at a
time (at present, Pyomo does not support any solvers that can be given
more than one objective). If both ``model.obj1`` and ``model.obj2`` have
been declared using ``Objective``, then one can ensure that
``model.obj2`` is passed to the solver as shown in this simple example:

.. doctest::

   >>> model = pyo.ConcreteModel()
   >>> model.obj1 = pyo.Objective(expr = 0)
   >>> model.obj2 = pyo.Objective(expr = 0)

   >>> model.obj1.deactivate()
   >>> model.obj2.activate()

For abstract models this would be done prior to instantiation or else
the ``activate`` and ``deactivate`` calls would be on the instance
rather than the model.

Activating and Deactivating Constraints
---------------------------------------

Constraints can be temporarily disabled using the ``deactivate()`` method.
When the model is sent to a solver inactive constraints are not included. 
Disabled constraints can be re-enabled using the ``activate()`` method.

.. doctest::

   >>> model = pyo.ConcreteModel()
   >>> model.v = pyo.Var()
   >>> model.con = pyo.Constraint(expr=model.v**2 + model.v >= 3)
   >>> model.con.deactivate()
   >>> model.con.activate()

Indexed constraints can be deactivated/activated as a whole or by 
individual index:

.. doctest::

   >>> model = pyo.ConcreteModel()
   >>> model.s = pyo.Set(initialize=[1,2,3])
   >>> model.v = pyo.Var(model.s)
   >>> def _con(m, s):
   ...    return m.v[s]**2 + m.v[s] >= 3
   >>> model.con = pyo.Constraint(model.s, rule=_con)
   >>> model.con.deactivate()   # Deactivate all indices
   >>> model.con[1].activate()  # Activate single index
