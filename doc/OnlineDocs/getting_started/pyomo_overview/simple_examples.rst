Simple Models
=============

A Simple Concrete Pyomo Model
*****************************

It is possible to get the same flexible behavior from models
declared to be abstract and models declared to be concrete in Pyomo;
however, we will focus on a straightforward concrete example here where
the data is hard-wired into the model file. Python programmers will
quickly realize that the data could have come from other sources.

Given the following model from the previous section:

.. math::
   :nowrap:

   \[\begin{array}{ll}
     \min       & 2 x_1 + 3 x_2\\
     \mathrm{s.t.} & 3 x_1 + 4 x_2 \geq 1\\
                & x_1, x_2 \geq 0
   \end{array}\]

This can be implemented as a concrete model as follows:

.. testcode::

   import pyomo.environ as pyo

   model = pyo.ConcreteModel()

   model.x = pyo.Var([1,2], domain=pyo.NonNegativeReals)

   model.OBJ = pyo.Objective(expr = 2*model.x[1] + 3*model.x[2])

   model.Constraint1 = pyo.Constraint(expr = 3*model.x[1] + 4*model.x[2] >= 1)

Although rule functions can also be used to specify constraints and
objectives, in this example we use the ``expr`` option that is available
only in concrete models. This option gives a direct specification of the
expression.


A Simple Abstract Pyomo Model
*****************************
We repeat the abstract model from the previous section:

.. math::
   :nowrap:

    \[\begin{array}{lll}
     \min      & \sum_{j=1}^n c_j x_j &\\
     \mathrm{s.t.} & \sum_{j=1}^n a_{ij} x_j \geq b_i & \forall i = 1 \ldots m\\
               & x_j \geq 0 & \forall j = 1 \ldots n
    \end{array}\]

One way to implement this in Pyomo is as shown as follows:

.. testcode::

   import pyomo.environ as pyo

   model = pyo.AbstractModel()

   model.m = pyo.Param(within=pyo.NonNegativeIntegers)
   model.n = pyo.Param(within=pyo.NonNegativeIntegers)

   model.I = pyo.RangeSet(1, model.m)
   model.J = pyo.RangeSet(1, model.n)

   model.a = pyo.Param(model.I, model.J)
   model.b = pyo.Param(model.I)
   model.c = pyo.Param(model.J)

   # the next line declares a variable indexed by the set J
   model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)

   def obj_expression(m):
       return pyo.summation(m.c, m.x)

   model.OBJ = pyo.Objective(rule=obj_expression)

   def ax_constraint_rule(m, i):
       # return the expression for the constraint for i
       return sum(m.a[i,j] * m.x[j] for j in m.J) >= m.b[i]

   # the next line creates one constraint for each member of the set model.I
   model.AxbConstraint = pyo.Constraint(model.I, rule=ax_constraint_rule)

.. doctest::
   :hide:

   >>> # Create an instance to verify that the rules fire correctly
   >>> inst = model.create_instance('src/scripting/abstract1.dat')

.. note::

   Python is interpreted one line at a time.  A line continuation
   character, ``\`` (backslash), is used for Python statements that need to span
   multiple lines.  In Python, indentation has meaning and must be
   consistent. For example, lines inside a function definition must be
   indented and the end of the indentation is used by Python to signal
   the end of the definition.

We will now examine the lines in this example.  
The first import line is required in every Pyomo model. Its purpose
is to make the symbols used by Pyomo known to Python.

.. testcode::

   import pyomo.environ as pyo

The declaration of a model is also required. The use of the name ``model``
is not required. Almost any name could be used, but we will use the name
``model`` in most of our examples. In this example, we are declaring
that it will be an abstract model.

.. testcode::

   model = pyo.AbstractModel()

We declare the parameters :math:`m` and :math:`n` using the Pyomo
:class:`Param` component. This component can take a variety of arguments; this
example illustrates use of the ``within`` option that is used by Pyomo
to validate the data value that is assigned to the parameter. If this
option were not given, then Pyomo would not object to any type of data
being assigned to these parameters. As it is, assignment of a value that
is not a non-negative integer will result in an error.

.. testcode::

   model.m = pyo.Param(within=pyo.NonNegativeIntegers)
   model.n = pyo.Param(within=pyo.NonNegativeIntegers)

Although not required, it is convenient to define index sets. In this
example we use the :class:`RangeSet` component to declare that the sets will
be a sequence of integers starting at 1 and ending at a value specified
by the the parameters ``model.m`` and ``model.n``.

.. testcode::

   model.I = pyo.RangeSet(1, model.m)
   model.J = pyo.RangeSet(1, model.n)

The coefficient and right-hand-side data are defined as indexed
parameters. When sets are given as arguments to the :class:`Param` component,
they indicate that the set will index the parameter.

.. testcode::

   model.a = pyo.Param(model.I, model.J)
   model.b = pyo.Param(model.I)
   model.c = pyo.Param(model.J)

The next line that is interpreted by Python as part of the model
declares the variable :math:`x`. The first argument to the :class:`Var`
component is a set, so it is defined as an index set for the variable. In
this case the variable has only one index set, but multiple sets could
be used as was the case for the declaration of the parameter
``model.a``. The second argument specifies a domain for the
variable. This information is part of the model and will passed to the
solver when data is provided and the model is solved. Specification of
the ``NonNegativeReals`` domain implements the requirement that the
variables be greater than or equal to zero.

.. testcode::

   # the next line declares a variable indexed by the set J
   model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)

.. note::

   In Python, and therefore in Pyomo, any text after pound sign is
   considered to be a comment.

In abstract models, Pyomo expressions are usually provided to objective
and constraint declarations via a function defined with a
Python ``def`` statement. The ``def`` statement establishes a name for a
function along with its arguments. When Pyomo uses a function to get
objective or constraint expressions, it always passes in the
model (i.e., itself) as the the first argument so the model is always
the first formal argument when declaring such functions in Pyomo.
Additional arguments, if needed, follow. Since summation is an extremely
common part of optimization models, Pyomo provides a flexible function
to accommodate it. When given two arguments, the :func:`summation()` function
returns an expression for the sum of the product of the two arguments
over their indexes. This only works, of course, if the two arguments
have the same indexes. If it is given only one argument it returns an
expression for the sum over all indexes of that argument. So in this
example, when :func:`summation` is passed the arguments ``m.c, m.x``
it returns an internal representation of the expression
:math:`\sum_{j=1}^{n}c_{j} x_{j}`.

.. testcode::

   def obj_expression(m):
       return pyo.summation(m.c, m.x)

To declare an objective function, the Pyomo component called
:class:`Objective` is used. The ``rule`` argument gives the name of a
function that returns the objective expression. The default *sense* is
minimization. For maximization, the ``sense=pyo.maximize`` argument must be
used. The name that is declared, which is ``OBJ`` in this case, appears
in some reports and can be almost any name.

.. testcode::

   model.OBJ = pyo.Objective(rule=obj_expression)

Declaration of constraints is similar. A function is declared to generate
the constraint expression. In this case, there can be multiple
constraints of the same form because we index the constraints by
:math:`i` in the expression :math:`\sum_{j=1}^n a_{ij} x_j \geq b_i
\;\;\forall i = 1 \ldots m`, which states that we need a constraint for
each value of :math:`i` from one to :math:`m`. In order to parametrize
the expression by :math:`i` we include it as a formal parameter to the
function that declares the constraint expression. Technically, we could
have used anything for this argument, but that might be confusing. Using
an ``i`` for an :math:`i` seems sensible in this situation.

.. testcode::

   def ax_constraint_rule(m, i):
       # return the expression for the constraint for i
       return sum(m.a[i,j] * m.x[j] for j in m.J) >= m.b[i]

.. note::

   In Python, indexes are in square brackets and function arguments are
   in parentheses.

In order to declare constraints that use this expression, we use the
Pyomo :class:`Constraint` component that takes a variety of arguments. In this
case, our model specifies that we can have more than one constraint of
the same form and we have created a set, ``model.I``, over which these
constraints can be indexed so that is the first argument to the
constraint declaration. The next argument gives the rule that
will be used to generate expressions for the constraints. Taken as a
whole, this constraint declaration says that a list of constraints
indexed by the set ``model.I`` will be created and for each member of
``model.I``, the function ``ax_constraint_rule`` will be called and it
will be passed the model object as well as the member of ``model.I``

.. testcode::

   # the next line creates one constraint for each member of the set model.I
   model.AxbConstraint = pyo.Constraint(model.I, rule=ax_constraint_rule)

In the object oriented view of all of this, we would say that ``model``
object is a class instance of the :class:`AbstractModel` class, and
``model.J`` is a :class:`Set` object that is contained by this model.  Many
modeling components in Pyomo can be optionally specified as *indexed*
*components*: collections of components that are referenced using one or
more values.  In this example, the parameter ``model.c`` is indexed with
set ``model.J``.

In order to use this model, data must be given for the values of the
parameters. Here is one file that provides data (in AMPL "``.dat``" format).

.. doctest::
   :hide:

   >>> # Create an instance to verify that the rules fire correctly
   >>> inst = model.create_instance('src/scripting/abstract1.dat')

.. literalinclude:: /src/scripting/abstract1.dat
   :language: text

There are multiple formats that can be used to provide data to a Pyomo
model, but the AMPL format works well for our purposes because it
contains the names of the data elements together with the data. In AMPL
data files, text after a pound sign is treated as a comment. Lines
generally do not matter, but statements must be terminated with a
semi-colon.

For this particular data file, there is one constraint, so the value of
``model.m`` will be one and there are two variables (i.e., the vector
``model.x`` is two elements long) so the value of ``model.n`` will be
two. These two assignments are accomplished with standard
assignments. Notice that in AMPL format input, the name of the model is
omitted.

::

   param m := 1 ;
   param n := 2 ;

There is only one constraint, so only two values are needed for
``model.a``. When assigning values to arrays and vectors in AMPL format,
one way to do it is to give the index(es) and the the value. The line 1
2 4 causes ``model.a[1,2]`` to get the value
4. Since ``model.c`` has only one index, only one index value is needed
so, for example, the line 1 2 causes ``model.c[1]`` to get the
value 2. Line breaks generally do not matter in AMPL format data files,
so the assignment of the value for the single index of ``model.b`` is
given on one line since that is easy to read.

::

   param a :=
    1 1 3
    1 2 4
    ;

    param c:=
    1 2
    2 3
    ;

    param b := 1 1 ;

.. _abstract2.py:

.. _abstract2.dat:

Symbolic Index Sets
*******************

When working with Pyomo (or any other AML), it is convenient to write
abstract models in a somewhat more abstract way by using index sets that
contain strings rather than index sets that are implied by
:math:`1,\ldots,m` or the summation from 1 to :math:`n`. When this is
done, the size of the set is implied by the input, rather than specified
directly. Furthermore, the index entries may have no real order.  Often,
a mixture of integers and indexes and strings as indexes is needed in
the same model. To start with an illustration of general indexes,
consider a slightly different Pyomo implementation of the model we just
presented.

.. literalinclude:: /src/scripting/abstract2.py
   :language: python

To get the same instantiated model, the following data file can be used.

.. literalinclude:: /src/scripting/abstract2a.dat
   :language: none

However, this model can also be fed different data for problems of the
same general form using meaningful indexes.

.. literalinclude:: /src/scripting/abstract2.dat
   :language: none


Solving the Simple Examples
***************************

Pyomo supports modeling and scripting but does not install a solver
automatically. In order to solve a model, there must be a solver
installed on the computer to be used. If there is a solver, then the
``pyomo`` command can be used to solve a problem instance.

Suppose that the solver named glpk (also known as glpsol) is installed
on the computer.  Suppose further that an abstract model is in the file
named ``abstract1.py`` and a data file for it is in the file named
``abstract1.dat``. From the command prompt, with both files in the
current directory, a solution can be obtained with the command:

::

   pyomo solve abstract1.py abstract1.dat --solver=glpk

Since glpk is the default solver, there really is no need specify it so
the ``--solver`` option can be dropped.

.. note::

   There are two dashes before the command line option names such as
   ``solver``.

To continue the example, if CPLEX is installed then it can be listed as
the solver. The command to solve with CPLEX is

::

   pyomo solve abstract1.py abstract1.dat --solver=cplex

This yields the following output on the screen:

::

   [    0.00] Setting up Pyomo environment
   [    0.00] Applying Pyomo preprocessing actions
   [    0.07] Creating model
   [    0.15] Applying solver
   [    0.37] Processing results
   Number of solutions: 1
   Solution Information
   Gap: 0.0
   Status: optimal
   Function Value: 0.666666666667
   Solver results file: results.json
   [    0.39] Applying Pyomo postprocessing actions
   [    0.39] Pyomo Finished

The numbers in square brackets indicate how much time was required for
each step. Results are written to the file named ``results.json``, which
has a special structure that makes it useful for post-processing. To see
a summary of results written to the screen, use the ``--summary``
option:

::

   pyomo solve abstract1.py abstract1.dat --solver=cplex --summary

To see a list of Pyomo command line options, use:

::

   pyomo solve --help

.. note::

   There are two dashes before ``help``.

For a concrete model, no data file is specified on the Pyomo command line.
