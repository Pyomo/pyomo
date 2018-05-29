Expressions
===========

In this chapter, we use the word "expression" in two ways: first in the general
sense of the word and second to desribe a class of Pyomo objects that have
the name ``Expression`` as described in the subsection on expression objects.

Rules to Generate Expressions
-----------------------------

Both objectives and constraints make use of rules to generate
expressions. These are Python functions that return the appropriate
expression. These are first-class functions that can access
global data as well as data passed in, including the model object.

Operations on model elements results in expressions, which seems
natural in expressions like the constraints we have seen so far. It is also
possible to build up expressions. The following example illustrates this, along
with a reference to global Pyton data in the form of a Python variable called ``switch``:

.. literalinclude:: spyfiles/spy4Expressions_Buildup_expression_switch.spy
   :language: python

In this example, the constraint that is generated depends on the value
of the Python variable called ``switch``. If the value is 2 or greater, then
the constraint is ``summation(model.c, model.x) - model.d >= 0.5``; otherwise,
the ``model.d`` term is not present.

CAUTION: Because model elements result in expressions, not values, the following
does not work as expected in an abstract model!

.. literalinclude:: spyfiles/spy4Expressions_Abstract_wrong_usage.spy
   :language: python

The trouble is that ``model.d >= 2`` results in an expression, not its evaluated value. Instead use ``if value(model.d) >= 2``

NOTE: Pyomo supports non-linear expressions and can call non-linear solvers such as Ipopt.

.. _piecewise:

.. _abstract2piece.py:

Piecewise Linear Expressions
----------------------------

Pyomo has facilities to add piecewise constraints of the form y=f(x) for a variety
of forms of the function f.

The piecewise types other than SOS2, BIGM_SOS1,
BIGM_BIN are implement as described in the paper [Vielma_et_al]_.

There are two basic forms for the declaration of the constraint:

.. literalinclude:: spyfiles/spy4Expressions_Declare_piecewise_constraints.spy
   :language: python

where ``pwconst`` can be replaced by a name appropriate for the application. The choice depends on whether the x and y
variables are indexed. If so, they must have the same index sets and these sets are give as the first arguments.

Keywords:
*********

* pw_pts=\{\},[],()
          A dictionary of lists (where keys are the index set) or a single list (for
          the non-indexed case or when an identical set of breakpoints is
          used across all indices) defining the set of domain breakpoints for
          the piecewise linear function.

          NOTE: pw_pts is always required. These give the breakpoints for the piecewise function
          and are expected to fully span the bounds for the independent variable(s).

* pw_repn=<Option>
          Indicates the type of piecewise representation to use. This can have
          a major impact on solver performance.
          Options: (Default "SOS2")

          * "SOS2"       -  Standard representation using sos2 constraints.
          * "BIGM_BIN"   -  BigM constraints with binary variables. The theoretically tightest M values are automatically determined.
          * "BIGM_SOS1"  -  BigM constraints with sos1 variables. The theoretically tightest M values are automatically determined.
          * "DCC"        -  Disaggregated convex combination model.
          * "DLOG"       -  Logarithmic disaggregated convex combination model.
          * "CC"         -  Convex combination model.
          * "LOG"        -  Logarithmic branching convex combination.
          * "MC"         -  Multiple choice model.
          * "INC"        -  Incremental (delta) method.

          NOTE: Step functions are supported for all but the two BIGM options. Refer to the 'force_pw' option.

* pw_constr_type= <Option>
          Indicates the bound type of the piecewise function. Options:

          * "UB" - y variable is bounded above by piecewise function.
          * "LB" - y variable is bounded below by piecewise function.
          * "EQ" - y variable is equal to the piecewise function.

* f_rule=f(model,i,j,...,x), \{\}, [], () +
          An object that returns a numeric value that is the range value
          corresponding to each piecewise domain point. For functions, the first
          argument must be a Pyomo model. The last argument is the domain value at
          which the function evaluates (Not a Pyomo ``Var``). Intermediate arguments
          are the corresponding indices of the Piecewise component (if any).
          Otherwise, the object can be a dictionary of lists/tuples (with keys the
          same as the indexing set) or a singe list/tuple (when no indexing set is
          used or when all indices use an identical piecewise function).
          Examples:

          .. literalinclude:: spyfiles/spy4Expressions_f_rule_Function_examples.spy
             :language: python

* force_pw=True/False
          Using the given function rule and pw_pts, a check for convexity/concavity
          is implemented. If (1) the function is convex and the piecewise
          constraints are lower bounds or if (2) the function is concave and the
          piecewise constraints are upper bounds then the piecewise constraints
          will be substituted for linear constraints. Setting 'force_pw=True' will
          force the use of the original piecewise constraints even when one of these
          two cases applies.

* warning_tol=<float>
          To aid in debugging, a warning is printed when consecutive slopes of
          piecewise segments are within <warning_tol> of each other. Default=1e-8

* warn_domain_coverage=True/False
          Print a warning when the feasible region of the domain variable is not
          completely covered by the piecewise breakpoints. Default=True

* unbounded_domain_var=True/False
          Allow an unbounded or partially bounded Pyomo Var to be used as the
          domain variable. Default=False

          NOTE: This does not imply unbounded piecewise segments will be
          constructed. The outermost piecewise breakpoints will bound the
          domain variable at each index. However, the Var attributes
          .lb and .ub will not be modified.

Here is an example of an assignment to a Python dictionary variable that has keywords for
a picewise constraint:

.. literalinclude:: spyfiles/spy4Expressions_Keyword_assignment_example.spy
   :language: python

Here is a simple example based on the example given earlier in :ref:`abstract2.py`. In this
new example, the objective function is the sum of c times x to the fourth. In this
example, the keywords are passed directly to the ``Piecewise`` function without
being assigned to a dictionary variable.  The upper bound on the x variables was chosen whimsically just to make the example.
The important thing to note is that variables that are going to appear as the independent variable in a piecewise constraint must have bounds.

.. literalinclude:: examples/abstract2piece.py
   :language: python

A more advanced example is provided in abstract2piecebuild.py in :ref:`BuildAction`.

``Expression`` Objects
----------------------

Pyomo ``Expression`` objects are very similar to the ``Param`` component
(with ``mutable=True``) except that the underlying values can be numeric
constants or Pyomo expressions. Here's an illustration of expression
objects in an AbstractModel.  An expression object with an index set
that is the numbers 1, 2, 3 is created and initialized to be the model
variable x times the index. Later in the model file, just to
illustrate how to do it, the expression is changed but just for the
first index to be x squared.

.. literalinclude:: spyfiles/spy4Expressions_Expression_objects_illustration.spy
   :language: python

An alternative is to create Python functions that, potentially, manipulate
model objects. E.g., if you define a function

.. literalinclude:: spyfiles/spy4Expressions_Define_python_function.spy
   :language: python

You can call this function with or without Pyomo modeling components as the arguments. E.g., f(2,3) will return a number, whereas f(model.x, 3) will return a Pyomo expression due to operator overloading.

If you take this approach you should note that anywhere a Pyomo expression is used to generate another expression
(e.g., f(model.x, 3) + 5), the initial expression is always cloned so that the new generated expression is independent of the old.
For example:

.. literalinclude:: spyfiles/spy4Expressions_Generate_new_expression.spy
   :language: python

If you want to create an expression that is shared between other
expressions, you can use the ``Expression`` component.


.. rubric:: References

.. [Vielma_et_al] Vielma, J.P, Ahmed, S., Nemhauser, G., "Mixed-Integer Models for Non-separable Piecewise Linear Optimization: Unifying framework and Extensions," Operations Research 58, 2010. pp. 303-315.
