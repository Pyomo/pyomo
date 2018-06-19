Pyomo Overview
===============

Mathematical Modeling
---------------------

This chapter provides an introduction to Pyomo: Python Optimization Modeling Objects.
A more complete description is contained in the [PyomoBookII]_ book. Pyomo
supports the formulation and analysis of mathematical models for complex
optimization applications.  This capability is commonly associated with
commerically available algebraic modeling languages (AMLs) such as AMPL [AMPL]_, AIMMS [AIMMS]_,
and GAMS [GAMS]_.  Pyomo's modeling objects are embedded within Python, a
full-featured, high-level programming language that contains a rich set of
supporting libraries.

Modeling is a fundamental process in many aspects of scientific research,
engineering and business.  Modeling involves the formulation of a simplified
representation of a system or real-world object.  Thus, modeling tools like
Pyomo can be used in a variety of ways:

.. autosummary::
   :nosignatures:

- *Explain phenomena* that arise in a system,

- *Make predictions* about future states of a system,

- *Assess key factors* that influence phenomena in a system,

- *Identify extreme states* in a system, that might represent worst-case scenarios or minimal cost plans, and

- *Analyze trade-offs* to support human decision makers.

Mathematical models represent system knowledge with a formalized language.
The following mathematical concepts are central to modern
modeling activities:

variables
*********
    Variables represent unknown or changing parts of a model (e.g., whether or not to make a decision, or the characteristic of a system outcome). The values taken by the variables are often referred to as a *solution* and are usually an output of the optimization process.

parameters
**********
    Parameters represents the data that must be supplied to perform the optimization. In fact, in some settings the word *data* is used in place of the word *parameters*.

relations
*********
    These are equations, inequalities or other mathematical relationships that define how different parts of a model are connected to each other.

goals
*****
    These are functions that reflect goals and objectives for the system being modeled.

The widespread availability of computing resources has made the
numerical analysis of mathematical models a commonplace activity.
Without a modeling language, the process of setting up input files,
executing a solver and extracting the final results from the solver
output is tedious and error-prone.  This difficulty is compounded
in complex, large-scale real-world applications which are difficult
to debug when errors occur.  Additionally, there are many different
formats used by optimization software packages, and few formats are
recognized by many optimizers.  Thus the application of multiple
optimization solvers to analyze a model introduces additional
complexities.

Pyomo is an AML that extends Python to include objects for mathematical modeling.
Hart et al. [PyomoBookI]_, [PyomoBookII]_, and [PyomoJournal]_ compare Pyomo with other AMLs.  Although many
good AMLs have been developed for optimization models, the following are motivating
factors for the development of Pyomo:

Open Source
***********
  Pyomo is developed within Pyomo's open source project to promote transparency of the modeling framework and encourage community development of Pyomo capabilities.

Customizable Capability
***********************
  Pyomo supports a customizable capability through the extensive use of plug-ins to modularize software components.

Solver Integration
******************
  Pyomo models can be optimized with solvers that are written either in Python or in compiled, low-level languages.

Programming Language
********************
  Pyomo leverages a high-level programming language, which has several advantages over custom AMLs:  a very robust language, extensive documentation, a rich set of standard libraries, support for modern programming features like classes and functions, and portability to many platforms.

Overview of Modeling Components and Processes
---------------------------------------------

Pyomo supports an object-oriented design for the definition of
optimization models.  The basic steps of a simple modeling process
are:

.. autosummary::
   :nosignatures:

* Create model and declare components
* Instantiate the model
* Apply solver
* Interrogate solver results

In practice, these steps may be applied repeatedly with different
data or with different constraints applied to the model.  However,
we focus on this simple modeling process to illustrate different
strategies for modeling with Pyomo.

A Pyomo *model* consists of a collection of
modeling *components* that define different aspects of the model.
Pyomo includes the modeling components that are commonly
supported by modern AMLs:  index sets, symbolic parameters, decision
variables, objectives, and constraints.
These modeling components are defined in Pyomo through the following Python classes:

Set
***
  set data that is used to define a model instance

Param
*****
  parameter data that is used to define a model instance

Var
***
  decision variables in a model

Objective
*********
  expressions that are minimized or maximized in a model

Constraint
**********
  constraint expressions that impose restrictions on variable values in a model

Abstract Versus Concrete Models
-------------------------------

A mathematical model can be defined using symbols that represent data values.
For example, the following equations represent a linear program
(LP) to find optimal values for the vector :math:`x` with parameters :math:`n` and :math:`b`, and parameter vectors :math:`a` and :math:`c`:

.. math::
   :nowrap:

    \begin{array}{lll}
    \min       & \sum_{j=1}^n c_j x_j &\\
     \mathrm{s.t.} & \sum_{j=1}^n a_{ij} x_j \geq b_i & \forall i = 1 \ldots m\\
               & x_j \geq 0 & \forall j = 1 \ldots n
     \end{array}

NOTE: As a convenience, we use the symbol :math:`\forall` to mean "for all" or "for each."

We call this an *abstract* or *symbolic* mathematical model since it relies on
unspecified parameter values.  Data values can be used to specify a *model instance*.
The ``AbstractModel`` class provides a context for defining and initializing abstract
optimization models in Pyomo when the data values will be supplied at the time a solution
is to be obtained.

In many contexts, a mathematical model can and should be directly defined
with the data values supplied at the time of the model definition.
We call these *concrete* mathematical models.
For example, the following LP model is a concrete instance of the previous abstract model:

.. math::
   :nowrap:

    \begin{array}{ll}
    \min       & 2 x_1 + 3 x_2\\
     \mathrm{s.t.} & 3 x_1 + 4 x_2 \geq 1\\
               & x_1, x_2 \geq 0
    \end{array}

The ``ConcreteModel`` class is used to define concrete optimization models in Pyomo.

NOTE: Python programmers will probably prefer to write concrete
models, while users of some other algebraic modeling languages may
tend to prefer to write abstract models.  The choice is largely a
matter of taste; some applications may be a little more
straightforward using one or the other.

A Simple Abstract Pyomo Model
-----------------------------
We repeat the abstract model already given:

.. math::
   :nowrap:

    \begin{array}{lll}
    \min       & \sum_{j=1}^n c_j x_j &\\
     \mathrm{s.t.} & \sum_{j=1}^n a_{ij} x_j \geq b_i & \forall i = 1 \ldots m\\
               & x_j \geq 0 & \forall j = 1 \ldots n
    \end{array}

One way to implement this in Pyomo is as shown as follows:

.. literalinclude:: spyfiles/abstract1.spy
   :language: python

NOTE: Python is interpreted one line at a time.  A line continuation
character, backslash, is used for Python statements that need to
span multiple lines.  In Python, indentation has meaning and must
be consistent. For example, lines inside a function definition must
be indented and the end of the indentation is used by Python to
signal the end of the definition.

We will now examine the lines in this example.  The first import
line is used to ensure that ``int`` or ``long`` division arguments are
converted to floating point values before division is performed.

>>> from __future__ import division

In Python versions before 3.0, division returns the floor of the
mathematical result of division if arguments are ``int`` or ``long``.
This import line avoids unexpected behavior when developing
mathematical models with integer values.

The next import line
that is required in every Pyomo model. Its purpose is to make the symbols used by Pyomo known to Python.

>>> from pyomo.environ import *

The declaration of a model is also required. The use of the name
`model` is not required. Almost any name could be used, but we will
use the name `model` most of the time in this book. In this example,
we are declaring that it will be an abstract model.

.. literalinclude:: spyfiles/abstract1_Declare_abstract_model.spy
   :language: python

We declare the parameters :math:`m` and :math:`n` using the Pyomo ``Param`` function. This function can take a
variety of arguments; this example illustrates use of the ``within`` option that is used by Pyomo to validate the
data value that is assigned to the parameter. If this option were not given, then Pyomo would not object to any type of data being
assigned to these parameters. As it is, assignment of a value that is not a non-negative integer will result in an error.

.. literalinclude:: spyfiles/abstract1_Declare_param_within.spy
   :language: python

Although not required, it is convenient to define index sets. In this example we use the ``RangeSet`` function to
declare that the sets will be a sequence of integers starting at 1 and ending at a value specified by the the parameters
``model.m`` and ``model.n``.

.. literalinclude:: spyfiles/abstract1_Define_indexsets.spy
   :language: python

The coefficient and right-hand-side data are defined as indexed parameters. When sets are given as arguments to the
``Param`` function, they indicate that the set will index the parameter.

.. literalinclude:: spyfiles/abstract1_Define_indexed_parameters.spy
   :language: python

NOTE: In Python, and therefore in Pyomo, any text after pound sign is considered to be a comment.

The next line that is
interpreted by Python as part of the model declares the variable :math:`x`. The first argument to the
``Var`` function is a set, so it is defined as an index set for the variable. In this case the variable
has only one index set, but multiple sets could be used as was the case for the declaration of the
parameter ``model.a``. The second argument specifies a domain for the variable. This information is part of the model
and will passed to the solver when data is provided and the model is solved. Specification of the
``NonNegativeReals`` domain implements the requirement that the variables be greater than or equal to zero.

.. literalinclude:: spyfiles/abstract1_Define_variable.spy
   :language: python

In abstract models, Pyomo expressions are usually provided to objective function and constraint declarations via a function
defined with a
Python ``def`` statement. The ``def`` statement establishes a name for a function along with its arguments. When
Pyomo uses a function to get objective function or constraint expressions, it always passes in the model (i.e., itself) as the
the first argument so the model is always the first formal argument when declaring such functions in Pyomo.
Additional arguments, if needed, follow. Since summation is an extremely common part of optimization models,
Pyomo provides a flexible function to
accommodate it. When given two arguments, the ``summation`` function returns an expression for the sum of the
product of the two arguments over their indexes. This
only works, of course, if the two arguments have the same indexes. If it is given only one argument it returns an expression for
the sum over all indexes of that argument. So in this example, when ``summation`` is passed the arguments ``model.c, model.x`` it returns an internal representation of the expression :math:`\sum_{j=1}^{n}c_{j} x_{j}`.

.. literalinclude:: spyfiles/abstract1_Define_objective_expression.spy
   :language: python

To declare an objective function, the Pyomo function called ``Objective`` is used. The ``rule`` argument
gives the name of a function that returns the expression to be used. The default *sense* is minimization. For
maximization, the ``sense=maximize`` argument must be used. The name that is declared, which is ``OBJ`` in this
case, appears in some reports and can be almost any name.

.. literalinclude:: spyfiles/abstract1_Declare_objective_function.spy
   :language: python

Declaration of constraints is similar. A function is declared to
deliver the constraint expression. In this case, there can be multiple
constraints of the same form because we index the constraints by
:math:`i` in the expression :math:`\sum_{j=1}^n a_{ij} x_j
\geq b_i \;\;\forall i = 1 \ldots m`, which states that we need a
constraint for each value of :math:`i` from one to
:math:`m`. In order to parametrize the expression by
:math:`i` we include it as a formal parameter to the function
that declares the constraint expression. Technically, we could have
used anything for this argument, but that might be confusing. Using an
``i`` for an :math:`i` seems sensible in this situation.

.. literalinclude:: spyfiles/abstract1_Define_constraints_expression.spy
   :language: python

NOTE: In Python, indexes are in square brackets and function arguments are in parentheses.

In order to declare constraints that use this expression, we use the Pyomo ``Constraint`` function
that takes a variety of arguments. In this case, our model specifies that we can have more than one constraint
of the same form and we have created a set, ``model.I``, over which these constraints can be indexed so
that is the first argument to the constraint declaration function. The next argument gives the rule that will be used to generate
expressions for the constraints. Taken as a whole, this constraint declaration says that a list of constraints indexed
by the set ``model.I`` will be created and for each member of ``model.I``, the function ``ax_constraint_rule`` will be
called and it will be passed the model object as well as the member of ``model.I``

.. literalinclude:: spyfiles/abstract1_Declare_constraints.spy
   :language: python

In the object oriented view of all of this, we would say that ``model`` object is a class instance of the ``AbstractModel`` class, and ``model.J`` is a
``Set`` object that is contained by this model.
Many modeling components in Pyomo can be optionally specified as *indexed* *components*:
collections of components that are referenced using one or more values.
In this example, the parameter ``model.c`` is indexed
with set ``model.J``.

In order to use this model, data must be given for the values of the parameters. Here is one file that
provides data.

.. literalinclude:: examples/abstract1.dat

There are multiple formats that can be used to provide data to a Pyomo model, but the AMPL format works well
for our purposes because it contains the names of the data elements together with the data. In AMPL data files,
text after a pound sign is treated as a comment. Lines generally do not matter, but statements must be terminated
with a semi-colon.

For this particular data file, there is one constraint, so the value of ``model.m`` will be one and there are two
variables (i.e., the vector ``model.x`` is two elements long) so the value of ``model.n`` will be two. These
two assignments are accomplished with standard assignments. Notice that in AMPL format input, the name of the model is
omitted.

::

   param m := 1 ;
   param n := 2 ;

There is only one constraint, so only two values are needed for ``model.a``. When assigning values to arrays and vectors in
AMPL format, one way to do it is to give the index(es) and the the value. The line 1 2 4 causes ``model.a[1,2]`` to get the value
4. Since ``model.c`` has only one index, only one index value is needed so, for example, the line 1 2 causes ``model.c[1]`` to
get the value 2. Line breaks generally do not matter in AMPL format data files, so the assignment of the value for the single index of
``model.b`` is given on one line since that is easy to read.

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

When working with Pyomo (or any other AML), it is convenient to write abstract models in a somewhat more
abstract way by using index sets that contain strings rather than index sets that are implied by
:math:`1,\ldots,m` or the summation from 1 to :math:`n`. When this is done, the size of the set
is implied by the input, rather than specified directly. Furthermore, the index entries may have no real order.
Often, a mixture of integers and indexes and strings as indexes is needed in the same model. To start with
an illustration of general indexes, consider a slightly different Pyomo implementation of the model we just presented.

.. literalinclude:: examples/abstract2.py
   :language: python

To get the same instantiated model, the following data file can be used.

.. literalinclude:: examples/abstract2a.dat

However, this model can also be fed different data for problems of the same general form using meaningful indexes.

.. literalinclude:: examples/abstract2.dat

A Simple Concrete Pyomo Model
*****************************

It is possible to get nearly the same flexible behavior from models declared to be abstract and models
declared to be concrete in Pyomo; however, we will focus on a straightforward concrete example here where
the data is hard-wired into the model file. Python programmers will quickly realize that the
data could have come from other sources.

We repeat the concrete model already given:

.. math::
   :nowrap:

    \begin{array}{ll}
     \min       & 2 x_1 + 3 x_2\\
     \mathrm{s.t.} & 3 x_1 + 4 x_2 \geq 1\\
                & x_1, x_2 \geq 0
     \end{array}

This is implemented as a concrete model as follows:

.. literalinclude:: examples/concrete1.py

Although rule functions can also be used to specify constraints and objectives, in this example we use the ``expr`` option that is available only in concrete models. This option gives a direct specification of the expression.

Solving the Simple Examples
***************************

Pyomo supports modeling and scripting but does not install a solver
automatically. In order to solve a model, there must be a solver installed
on the computer to be used. If there is a solver, then the ``pyomo``
command can be used to solve a problem instance.

Suppose that the solver named glpk (also known as glpsol) is installed on the computer.
Suppose further that an abstract model is in the file named ``abstract1.py`` and a data file
for it is in the file named ``abstract1.dat``. From the command prompt, with both files in the
current directory, a solution can be obtained with the command:

::

   pyomo solve abstract1.py abstract1.dat --solver=glpk

Since glpk is the default solver, there really is no need specify it so the
``--solver`` option can be dropped.

NOTE: There are two dashes before the command line option names such as
``solver``.

To continue the example, if CPLEX is installed then it can be listed as the solver. The command to solve with
CPLEX is

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

The numbers in square brackets indicate how much time was required for each step. Results are written to the file named ``results.json``, which
has a special structure that makes it useful for post-processing. To see a summary of results written to the screen, use the
``--summary`` option:

::
pyomo solve abstract1.py abstract1.dat --solver=cplex --summary

To see a list of Pyomo command line options, use:

::

   pyomo solve --help

NOTE: There are two dashes before ``help``.

For a concrete model, no data file is specified on the Pyomo command line.

