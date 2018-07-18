
Using Black-Box Optimizers with Pyomo.Opt
=========================================

Many optimization software packages contain *black-box* optimizers,
which perform optimization without using detailed knowledge of the
structure of an optimization problem.  Thus, black-box optimizers
require a generic interface for optimization problems that defines
key features of problems, like objectives and constraints.

The ``pyomo.opt`` package contains the ``pyomo.opt.blackbox`` subpackage,
which provides facilities for (a) integrating Pyomo solvers with
blackbox optimization applications and (b) wrapping Pyomo models
for use by external blackbox optimizers.  We illustrate these
capabilities in this chapter with simple examples that illustrate
the use of ``pyomo.opt.blackbox``.



Defining and Optimizing Simple Black-Box Applications
-----------------------------------------------------

Many black-box optimizers interact with an optimization problem by
executing a separate process that computes properties of the
optimization problem.  This process typically reads an input file
that defines the requested properties and writes an output file
that contains the computed values.  Unfortunately, no standards
have emerged for black-box optimizers that interact with problems
in this manner.  Thus, different file formats are used by different
optimizer software packages.


Defining an Optimization Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``pyomo.opt.blackbox`` package provides several Python classes
for optimization problems that coordinates file I/O for the user
and simplifies the definition of simple black-box problems.  The
``RealOptProblem`` class provides a generic interface for continuous
optimization problems (i.e. with real variables).  The
following example defines a simple continuous optimization problem:

.. literalinclude:: scripts_examples/RealProblem1@prob.py
   :language: python

This problem is equivalent to the following problem definition:

.. math::
   :nowrap:

   \begin{array}{lll}
   \min            & x_0 - x_1 + (x_2 - 1.5)^2 + (x_3+2)^4 & \\
    \mathrm{s.t.}   & 0 \leq x_0 & \\
                & -1 \leq x_1 \leq 0 & \\
                & 0 \leq x_2 \leq 2 & \\
                & x_3 \leq -1 & \\
   \end{array}

Note that the problem class does *not* specify the sense of the
optimization problem.  These problem classes are not a complete
specification of an optimization problem.  Rather, an instance of
a problem class can compute information about the problem that is
used during optimization.

Similarly, the ``MixedIntOptProblem`` class provides a generic interface
for mixed-integer optimization problems, which may contain real
variables, integer variables and binary variables.  The following example defines
a simple mixed-integer optimization problem:

.. literalinclude:: scripts_examples/MIProblem1@prob.py
   :language: python

This problem is equivalent to the following problem definition:

.. math::
   :nowrap:

   \begin{array}{lll}
   \min            & \sum_{i=1}^4 (x_i-1)^2 + \sum_{i=1}^3 (y_i+1)^2 + \sum_{i=1}^2 z_i & \\
    \mathrm{s.t.}   & 0 \leq x_i \leq 2 & \\
                & -2 \leq y_i \leq 0 & \\
                & z_i \in \{0,1\} & \\
   \end{array}


Optimizating with Coliny Solvers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Coliny`` software library supports interfaces to a variety of
black-box optimizers <Coliny>.  The ``coliny`` executable reads an
XML specification of the optimization problem and solver, as well
as a specification of how the optimizer is applied.  Consider the
following XML specification:

.. literalinclude:: scripts_examples/RealProblem1.xml
   :language: xml

This XML specification defines a ``MINLP0`` problem, which indicated
that this is a mixed-integer problem that supports zero-order
derivatives (i.e. no derivatives).  This problem has four real
variables with lower and upper bounds specified.  The problem values
are computed with the ``RealProblem1.py`` command-line, which defines
and uses the ``RealProblem1`` class defined above:

.. literalinclude:: scripts_examples/RealProblem1@.py
   :language: python

Note that this command is a Python script that includes the shebang
character sequence on the first line.  On Linux and Unix systems,
this line indicates that this is a script that is executed using
the ``python`` command that is found in the user environment.  Thus,
this example assumes that the ``python`` command has ``pyomo.opt``
installed.  Since multiple versions of Python can be installed on
a single computer, the XML ``Command`` element may need to be defined
with an explicitly Python version.  For example, if Python 2.6 is
installed in ``/usr/local`` with ``pyomo.opt``, then the ``Command``
element would look like:

::

   <Command>/usr/local/bin/python26 RealProblem1.py</Command>

Additionally, the duplication of bounds information between
``RealProblem1.py`` and ``RealProblem1.xml`` is not strictly necessary
in this example.  The bounds information in ``RealProblem1.py`` is
used in the ``validate`` method to verify that the point being evaluated
is consistent with the bounds information.  We can generally assume
that the Coliny solver will only evaluate feasible points, so a
simpler problem definition can be used:

.. literalinclude:: scripts_examples/RealProblem2@.py
   :language: python

The last two lines of ``RealProblem1.py`` create a problem instance
and then call the ``main`` method to parse the command-line arguments.
This script has the following command-line syntax:

::

  RealProblem1.py <input-file> <output-file>

The first argument is the name of an XML input file, and the second
argument is the name of an XML output file.  The optimization problem
class manages the parsing of the input and generation of the output
file.  For example, consider the following input file:

.. literalinclude:: scripts_examples/RealProblem1_request.xml
   :language: xml

The ``RealProblem1.py`` script creates the following output file:

.. literalinclude:: scripts_examples/RealProblem1_results.xml
   :language: xml


Diving Deeper
-------------

The previous section provided an overview of the how the
``pyomo.opt.blackbox`` package supports the definition of optimization
problems that are solved with black-box optimizers.  In this section
we provide more detail about how the Python problem class can be
customized, as well as details about the XML file format used to
communicate with Coliny optimizers.  The Dakota User Manual <Dakota>
provides documentation of the file format of the input and output
files used with Dakota optimizers.

The following table summarizes the methods of the
``OptProblem`` class that a user is likely to either use or redefine
when declaring a subclass.  The ``MixedIntOptProblem`` class is a
convenient base class for the problems solved by most black-box
optimizers, and this class provides the definition of the ``main``,
``create_point`` and ``validate`` methods.  However, any of the remaining
methods may need to be defined, depending on the problem.


===========================     ========================================================================================
Method                          Description
===========================     ========================================================================================
__init__                        The constructor, which may be redefined to specify problem properties.
main                            Method that processes command-line options to create a results file from an input file.
create_point                    Create an instances of the class that defines a point in the search domain.
function_value                  Returns the value of the objective function.
function_values                 Returns a list of objective function values.
gradient                        Returns a list that represents the gradient vector at the given point.
hessian                         Returns a Hessian matrix.
nonlinear_constraint_values     Returns a list of values for the constraint functions.
jacobian                        Returns a Jacobian matrix.
validate                        Returns ``True`` if the given pointis feasible, and ``False`` otherwise.
===========================     ========================================================================================

The following detailed example illustrates the use of all of these methods in a
simple application:

.. literalinclude:: scripts_examples/RealProblem3@prob.py
   :language: python

The ``response_types`` attribute defined in the constructor specifies the type of information that
this class can compute.  For example, consider the following input XML file:

.. literalinclude:: scripts_examples/RealProblem3_request.xml
   :language: xml

This input file requests that the class compute all of the response values, and thus the following output is generated:

.. literalinclude:: scripts_examples/RealProblem3_results.xml
   :language: xml

Note that the values for Jacobian and Hessian matrices are represented in a sparse manner.  Currently, these are represented with a list of tuple values, though a sparse matrix representation might be supported in the future.
