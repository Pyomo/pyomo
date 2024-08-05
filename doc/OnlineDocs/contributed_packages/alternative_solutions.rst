###############################################
Generating Alternative (Near-)Optimal Solutions
###############################################

Optimization solvers are generally designed to return a feasible solution
to the user. However, there are many applications where a users needs
more context than this result. For example,

* alternative solutions can support an assessment of trade-offs between competing objectives;

* if the optimization formulation may be inaccurate or untrustworthy, then comparisons amongst alternative solutions provides additional insights into the reliability of these model predictions; or

* the user may have unexpressed objectives or constraints, which only are realized in later stages of model analysis.

The *alternative-solutions library* provides a variety of functions that
can be used to generate optimal or near-optimal solutions for a pyomo
model. Conceptually, these functions are like pyomo solvers. They can
be configured with solver names and options, and they return a list of
solutions for the pyomo model. However, these functions are independent
of pyomo's solver interface because they return a custom solution object.

The following functions are defined in the alternative-solutions library:

* ``enumerate_binary_solutions``

    * Finds alternative optimal solutions for a binary problem using no-good cuts.

* ``enumerate_linear_solutions``

    * Finds alternative optimal solutions a (mixed-integer) linear program.

* ``enumerate_linear_solutions_soln_pool``

    * Finds alternative optimal solutions for a (mixed-binary) linear program using Gurobi's solution pool feature.

* ``gurobi_generate_solutions``

    * Finds alternative optimal solutions for discrete variables using Gurobi's built-in Solution Pool capability.

* ``obbt_analysis_bounds_and_solutions``

    * Calculates the bounds on each variable by solving a series of min and max optimization problems where each variable is used as the objective function. This can be applied to any class of problem supported by the selected solver.


Usage Example
-------------

Many of functions in the alternative-solutions library have similar options, so we simply illustrate the ``enumerate_binary_solutions`` function.  We define a simple model whose feasible space is a isosceles right triangle. The optimal solutiosn fall along the hypotenuse, where :math:`x + y == 5`.  Alternative near-optimal feasible points have integer objective values ranging from 0 to 4.

.. doctest::

   >>> import pyomo.environ as pyo

   >>> m = pyo.ConcreteModel()
   >>> m.x = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, 5))
   >>> m.y = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, 5))
   >>> m.o = pyo.Objective(expr=m.x + m.y, sense=pyo.maximize)
   >>> m.c = pyo.Constraint(expr=m.x + m.y <= 5)

We can execute the ``enumerate_binary_solutions`` function to generate a list of ``Solution`` objects that represent alternative optimal solutions:

.. doctest::
   :skipif: not glpk_available

   >>> import pyomo.contrib.alternative_solutions as aos
   >>> solns = aos.enumerate_binary_solutions(m, num_solutions=100, solver="glpk")
   >>> assert len(solns) == 1

Each ``Solution`` object contains information about the objective and variables, and it includes various methods to access this information.  For example:

.. doctest::
   :skipif: not glpk_available

   >>> print(solns[0])
   {
       "fixed_variables": [],
       "objective": "o",
       "objective_value": 5.0,
       "solution": {
           "x": 5,
           "y": 0
       }
   }


Interface Documentation
-----------------------

.. currentmodule:: pyomo.contrib.alternative_solutions

.. autofunction:: enumerate_binary_solutions

.. autofunction:: enumerate_linear_solutions

.. autofunction:: enumerate_linear_solutions_soln_pool

.. autofunction:: gurobi_generate_solutions

.. autofunction:: obbt_analysis_bounds_and_solutions

.. autoclass:: Solution

