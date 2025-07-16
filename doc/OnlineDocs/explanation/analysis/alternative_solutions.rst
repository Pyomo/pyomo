###############################################
Generating Alternative (Near-)Optimal Solutions
###############################################

Optimization solvers are generally designed to return a feasible solution
to the user. However, there are many applications where a user needs
more context than this result. For example,

* alternative solutions can support an assessment of trade-offs between
  competing objectives;

* if the optimization formulation may be inaccurate or untrustworthy,
  then comparisons amongst alternative solutions provide additional
  insights into the reliability of these model predictions; or

* the user may have unexpressed objectives or constraints, which only
  are realized in later stages of model analysis.

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

    * Finds alternative optimal solutions for a (mixed-integer) linear program.

* ``enumerate_linear_solutions_soln_pool``

    * Finds alternative optimal solutions for a (mixed-binary) linear
      program using Gurobi's solution pool feature.

* ``gurobi_generate_solutions``

    * Finds alternative optimal solutions for discrete variables using
      Gurobi's built-in solution pool capability.

* ``obbt_analysis_bounds_and_solutions``

    * Calculates the bounds on each variable by solving a series of min
      and max optimization problems where each variable is used as the
      objective function. This can be applied to any class of problem
      supported by the selected solver.


Basic Usage Example
-------------------

Many of the functions in the alternative-solutions library have similar
options, so we simply illustrate the ``enumerate_binary_solutions``
function.  We define a simple knapsack example whose alternative
solutions have integer objective values ranging from 0 to 90.

.. doctest::

   >>> import pyomo.environ as pyo

   >>> values = [10, 40, 30, 50]
   >>> weights = [5, 4, 6, 3]
   >>> capacity = 10

   >>> m = pyo.ConcreteModel()
   >>> m.x = pyo.Var(range(4), within=pyo.Binary)
   >>> m.o = pyo.Objective(expr=sum(values[i] * m.x[i] for i in range(4)), sense=pyo.maximize)
   >>> m.c = pyo.Constraint(expr=sum(weights[i] * m.x[i] for i in range(4)) <= capacity)

We can execute the ``enumerate_binary_solutions`` function to generate a
list of ``Solution`` objects that represent alternative optimal
solutions:

.. doctest::
   :skipif: not glpk_available

   >>> import pyomo.contrib.alternative_solutions as aos
   >>> solns = aos.enumerate_binary_solutions(m, num_solutions=100, solver="glpk")
   >>> assert len(solns) == 10

Each ``Solution`` object contains information about the objective and
variables, and it includes various methods to access this information.
For example:

.. doctest::
   :skipif: not glpk_available

   >>> print(solns[0])
   {
       "fixed_variables": [],
       "objective": "o",
       "objective_value": 90.0,
       "solution": {
           "x[0]": 0,
           "x[1]": 1,
           "x[2]": 0,
           "x[3]": 1
       }
   }


Gap Usage Example
-----------------

When we only want some of the solutions based off a tolerance away from
optimal, this can be done using the ``abs_opt_gap`` parameter. This is
shown in the following simple knapsack examples where the weights and
values are the same.

.. doctest::
   :skipif: not glpk_available

   >>> import pyomo.environ as pyo
   >>> import pyomo.contrib.alternative_solutions as aos

   >>> values = [10,9,2,1,1]
   >>> weights = [10,9,2,1,1]

   >>> K = len(values)
   >>> capacity = 12

   >>> m = pyo.ConcreteModel()
   >>> m.x = pyo.Var(range(K), within=pyo.Binary)
   >>> m.o = pyo.Objective(expr=sum(values[i] * m.x[i] for i in range(K)), sense=pyo.maximize)
   >>> m.c = pyo.Constraint(expr=sum(weights[i] * m.x[i] for i in range(K)) <= capacity)

   >>> solns = aos.enumerate_binary_solutions(m, num_solutions=10, solver="glpk", abs_opt_gap = 0.0)
   >>> assert(len(solns) == 4)

In this example, we only get the four ``Solution`` objects that have an
``objective_value`` of 12.  Note that while we wanted only those four
solutions with no optimality gap, using a gap of half the smallest value
(in this case .5) will return the same solutions and avoids any machine
precision issues.

.. doctest::
   :skipif: not glpk_available

   >>> import pyomo.environ as pyo
   >>> import pyomo.contrib.alternative_solutions as aos

   >>> values = [10,9,2,1,1]
   >>> weights = [10,9,2,1,1]

   >>> K = len(values)
   >>> capacity = 12

   >>> m = pyo.ConcreteModel()
   >>> m.x = pyo.Var(range(K), within=pyo.Binary)
   >>> m.o = pyo.Objective(expr=sum(values[i] * m.x[i] for i in range(K)), sense=pyo.maximize)
   >>> m.c = pyo.Constraint(expr=sum(weights[i] * m.x[i] for i in range(K)) <= capacity)

   >>> solns = aos.enumerate_binary_solutions(m, num_solutions=10, solver="glpk", abs_opt_gap = 0.5)
   >>> assert(len(solns) == 4)
   >>> for soln in sorted(solns, key=lambda s: str(s.get_variable_name_values())):
   ...     print(soln)
   {
       "fixed_variables": [],
       "objective": "o",
       "objective_value": 12.0,
       "solution": {
           "x[0]": 0,
           "x[1]": 1,
           "x[2]": 1,
           "x[3]": 0,
           "x[4]": 1
       }
   }
   {
       "fixed_variables": [],
       "objective": "o",
       "objective_value": 12.0,
       "solution": {
           "x[0]": 0,
           "x[1]": 1,
           "x[2]": 1,
           "x[3]": 1,
           "x[4]": 0
       }
   }
   {
       "fixed_variables": [],
       "objective": "o",
       "objective_value": 12.0,
       "solution": {
           "x[0]": 1,
           "x[1]": 0,
           "x[2]": 0,
           "x[3]": 1,
           "x[4]": 1
       }
   }
   {
       "fixed_variables": [],
       "objective": "o",
       "objective_value": 12.0,
       "solution": {
           "x[0]": 1,
           "x[1]": 0,
           "x[2]": 1,
           "x[3]": 0,
           "x[4]": 0
       }
   }


Interface Documentation
-----------------------

.. currentmodule:: pyomo.contrib.alternative_solutions

.. autofunction:: enumerate_binary_solutions
   :noindex:

.. autofunction:: enumerate_linear_solutions
   :noindex:

.. autofunction:: pyomo.contrib.alternative_solutions.lp_enum_solnpool.enumerate_linear_solutions_soln_pool
   :noindex:

.. autofunction:: gurobi_generate_solutions
   :noindex:

.. autofunction:: obbt_analysis_bounds_and_solutions
   :noindex:

.. autoclass:: Solution
   :noindex:

