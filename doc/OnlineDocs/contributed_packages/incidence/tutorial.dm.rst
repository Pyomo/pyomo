.. _incidence_tutorial_dm:

Debugging a structural singularity with the Dulmage-Mendelsohn partition
========================================================================

We start with some imports and by creating a Pyomo model we would like
to debug. Usually the model is much larger and more comlicated than this.
This particular system appeared when debugging a dynamic 1-D partial
differential-algebraic equation (PDAE) model representing a chemical looping
combustion reactor.

.. doctest::
   :skipif: not scipy_available or not networkx_available or not asl_available

   >>> import pyomo.environ as pyo
   >>> from pyomo.contrib.incidence_analysis import IncidenceGraphInterface

   >>> m = pyo.ConcreteModel()
   >>> m.components = pyo.Set(initialize=[1, 2, 3]) 
   >>> m.x = pyo.Var(m.components, initialize=1.0/3.0)
   >>> m.flow_comp = pyo.Var(m.components, initialize=10.0)
   >>> m.flow = pyo.Var(initialize=30.0)
   >>> m.density = pyo.Var(initialize=1.0)
   >>> m.sum_eqn = pyo.Constraint(
   ...     expr=sum(m.x[j] for j in m.components) - 1 == 0
   ... )
   >>> m.holdup_eqn = pyo.Constraint(m.components, expr={
   ...     j: m.x[j]*m.density - 1 == 0 for j in m.components
   ... })
   >>> m.density_eqn = pyo.Constraint(
   ...     expr=1/m.density - sum(1/m.x[j] for j in m.components) == 0
   ... )
   >>> m.flow_eqn = pyo.Constraint(m.components, expr={
   ...     j: m.x[j]*m.flow - m.flow_comp[j] == 0 for j in m.components
   ... })

To check this model for structural singularity, we apply the Dulmage-Mendelsohn
partition.

.. doctest::
   :skipif: not scipy_available or not networkx_available or not asl_available

   >>> igraph = IncidenceGraphInterface(m)
   >>> # Make sure we have a square system
   >>> print(len(igraph.variables))
   8
   >>> print(len(igraph.constraints))
   8
   >>> var_dmp, con_dmp = igraph.dulmage_mendelsohn()

If any variables or constraints are unmatched, the (Jacobian of the model)
is structurally singular.

.. doctest::
   :skipif: not scipy_available or not networkx_available or not asl_available

   >>> for var in var_dmp.unmatched:
   ...     print(f"  {var.name}")
   flow_comp[1]
   >>> for con in con_dmp.unmatched:
   ...     print(f"  {con.name}")
   density_eqn

This model has one unmatched constraint and one unmatched variable, so it is
structurally singular. However, the unmatched variable and constraint are not
unique. For example, ``flow_comp[2]`` could have been unmatched instead of
``flow_comp[1]``.

Unique subsets of variables and constraints that are useful when debugging a
structural singularity are the underconstrained and overconstrained subsystems.
The variables in the underconstrained subsystem are contained in the
``unmatched`` and ``underconstrained`` fields of the ``var_dmp`` named tuple,
while the constraints are contained in the ``underconstrained`` field of the
``con_dmp`` named tuple.
The variables in the overconstrained subsystem are contained in the
``overconstrained`` field of the ``var_dmp`` named tuple, while the constraints
are contained in the ``overconstrained`` and ``unmatched`` fields of the
``con_dmp`` named tuple.

We now construct the underconstrained and overconstrained subsystems:

.. doctest::
   :skipif: not scipy_available or not networkx_available or not asl_available

   >>> uc_var = var_dmp.unmatched + var_dmp.underconstrained
   >>> uc_con = con_dmp.underconstrained
   >>> oc_var = var_dmp.overconstrained
   >>> oc_con = con_dmp.overconstrained + con_dmp.unmatched

And display the variables and constraints contained in each:

.. code-block:: python

   >>> # Overconstrained subsystem
   >>> for var in oc_var:
   >>>     print(f"  {var.name}")
   x[1]
   density
   x[2]
   x[3]
   >>> for con in oc_con:
   >>>     print(f"  {con.name}")
   sum_eqn
   holdup_eqn[1]
   holdup_eqn[2]
   holdup_eqn[3]
   density_eqn

   >>> # Underconstrained subsystem
   >>> for var in uc_var:
   >>>     print(f"  {var.name}")
   flow_comp[1]
   flow
   flow_comp[2]
   flow_comp[3]
   >>> for con in uc_con:
   >>>     print(f"  {con.name}")
   flow_eqn[1]
   flow_eqn[2]
   flow_eqn[3]

At this point we must use our intuition about the system being model to
identify "what is causing" the singularity. Looking at the under and over-
constrained systems, it appears that we are missing an equation to calculate
``flow``, the total flow rate, and that ``density`` is over-specified as it
is computed by both the bulk density equation and one of the component density
equations.

With this knowledge, we can eventually figure out (a) that we need and equation
to calculate ``flow`` from density and (b) that our "bulk density equation"
is actually a *skeletal* density equation. Admittedly, this is difficult to
figure out without the full context behind this particular system.

The following code constructs a new version of the model and verifies that it
is structural singularity:

.. doctest::
   :skipif: not scipy_available or not networkx_available or not asl_available

   >>> import pyomo.environ as pyo
   >>> from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
   ... 
   >>> m = pyo.ConcreteModel()
   >>> m.components = pyo.Set(initialize=[1, 2, 3])
   >>> m.x = pyo.Var(m.components, initialize=1.0/3.0)
   >>> m.flow_comp = pyo.Var(m.components, initialize=10.0)
   >>> m.flow = pyo.Var(initialize=30.0)
   >>> m.dens_bulk = pyo.Var(initialize=1.0)
   >>> m.dens_skel = pyo.Var(initialize=1.0)
   >>> m.porosity = pyo.Var(initialize=0.25)
   >>> m.velocity = pyo.Param(initialize=1.0)
   >>> m.sum_eqn = pyo.Constraint(
   ...     expr=sum(m.x[j] for j in m.components) - 1 == 0
   ... )
   >>> m.holdup_eqn = pyo.Constraint(m.components, expr={
   ...     j: m.x[j]*m.dens_bulk - 1 == 0 for j in m.components
   ... })
   >>> m.dens_skel_eqn = pyo.Constraint(
   ...     expr=1/m.dens_skel - sum(1/m.x[j] for j in m.components) == 0
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

   >>> igraph = IncidenceGraphInterface(m, include_inequality=False)
   >>> print(len(igraph.variables))
   10
   >>> print(len(igraph.constraints))
   10
   >>> var_dmp, con_dmp = igraph.dulmage_mendelsohn()

   >>> # There are now no unmatched variables and equations
   >>> for var in var_dmp.unmatched:
   ...     print(f"  {var.name}")
   >>> for con in con_dmp.unmatched:
   ...     print(f"  {con.name}")
