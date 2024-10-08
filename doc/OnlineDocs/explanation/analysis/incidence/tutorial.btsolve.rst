Solving a square system with a block triangular decomposition
=============================================================

We start with imports. The key function from Incidence Analysis we will use is
``solve_strongly_connected_components``.

.. doctest::
   :skipif: not networkx_available or not scipy_available or not asl_available

   >>> import pyomo.environ as pyo 
   >>> from pyomo.contrib.incidence_analysis import (
   ...     solve_strongly_connected_components
   ... )

Now we construct the model we would like to solve. This is a model with the
same structure as the "fixed model" in :ref:`incidence_tutorial_dm`.

.. doctest::
   :skipif: not networkx_available or not scipy_available or not asl_available

   >>> m = pyo.ConcreteModel()
   >>> m.components = pyo.Set(initialize=[1, 2, 3]) 
   >>> m.x = pyo.Var(m.components, initialize=1.0/3.0)
   >>> m.flow_comp = pyo.Var(m.components, initialize=10.0)
   >>> m.flow = pyo.Var(initialize=30.0)
   >>> m.dens_bulk = pyo.Var(initialize=1.0)
   >>> m.dens_skel = pyo.Var(initialize=1.0)
   >>> m.porosity = pyo.Var(initialize=0.25)
   >>> m.velocity = pyo.Param(initialize=1.0)
   >>> m.holdup = pyo.Param(
   ...     m.components, initialize={j: 1.0+j/10.0 for j in m.components}
   ... )
   >>> m.sum_eqn = pyo.Constraint(
   ...     expr=sum(m.x[j] for j in m.components) - 1 == 0
   ... )
   >>> m.holdup_eqn = pyo.Constraint(m.components, expr={
   ...     j: m.x[j]*m.dens_bulk - m.holdup[j] == 0 for j in m.components
   ... })
   >>> m.dens_skel_eqn = pyo.Constraint(
   ...     expr=1/m.dens_skel - sum(1e-3/m.x[j] for j in m.components) == 0
   ... )
   >>> m.dens_bulk_eqn = pyo.Constraint(
   ...     expr=m.dens_bulk == (1 - m.porosity)*m.dens_skel
   ... )
   >>> m.flow_eqn = pyo.Constraint(m.components, expr={
   ...     j: m.x[j]*m.flow - m.flow_comp[j] == 0 for j in m.components
   ... })
   >>> m.flow_dens_eqn = pyo.Constraint(
   ...     expr=m.flow == m.velocity*m.dens_bulk
   ... )

Solving via a block triangular decomposition is useful in cases where the full
model does not converge when considered simultaneously by a Newton solver.
In this case, we specify a solver to use for the diagonal blocks and call
``solve_strongly_connected_components``.

.. doctest::
   :skipif: not networkx_available or not scipy_available or not asl_available

   >>> # Suppose a solve like this does not converge
   >>> # pyo.SolverFactory("scipy.fsolve").solve(m)

   >>> # We solve via block-triangular decomposition
   >>> solver = pyo.SolverFactory("scipy.fsolve")
   >>> res_list = solve_strongly_connected_components(m, solver=solver)

We can now display the variable values at the solution:

.. code-block:: python

   for var in m.component_objects(pyo.Var):
       var.pprint()
