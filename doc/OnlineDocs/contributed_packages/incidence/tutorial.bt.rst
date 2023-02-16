Debugging a numeric singularity using block triangularization
=============================================================

We start with some imports. To debug a *numeric* singularity, we will need
``PyomoNLP`` from :ref:`pynumero` to get the constraint Jacobian,
and will need NumPy to compute condition numbers.

.. doctest::
   :skipif: not scipy_available or not asl_available or not networkx_available

   >>> import pyomo.environ as pyo 
   >>> from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
   >>> from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
   >>> import numpy as np

We now build the model we would like to debug. Compared to the model in
:ref:`incidence_tutorial_dm`, we have converted the sum equation to use a sum
over component flow rates rather than a sum over mass fractions.

.. doctest::
   :skipif: not scipy_available or not asl_available or not networkx_available

   >>> m = pyo.ConcreteModel()
   >>> m.components = pyo.Set(initialize=[1, 2, 3]) 
   >>> m.x = pyo.Var(m.components, initialize=1.0/3.0)
   >>> m.flow_comp = pyo.Var(m.components, initialize=10.0)
   >>> m.flow = pyo.Var(initialize=30.0)
   >>> m.density = pyo.Var(initialize=1.0)
   >>> # This equation is new!
   >>> m.sum_flow_eqn = pyo.Constraint(
   ...     expr=sum(m.flow_comp[j] for j in m.components) == m.flow
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

We now construct the incidence graph and check unmatched variables and
constraints to validate structural nonsingularity.

.. doctest::
   :skipif: not scipy_available or not asl_available or not networkx_available

   >>> igraph = IncidenceGraphInterface(m, include_inequality=False)
   >>> var_dmp, con_dmp = igraph.dulmage_mendelsohn()
   >>> print(len(var_dmp.unmatched))
   0
   >>> print(len(con_dmp.unmatched))
   0

Our system is structurally nonsingular. Now we check whether we are numerically
nonsingular (well-conditioned) by checking the condition number.
Admittedly, deciding if a matrix is "singular" by looking at its condition
number is somewhat of an art. We might define "numerically singular" as having a
condition number greater than the inverse of machine precision (approximately
``1e16``), but poorly conditioned matrices can cause problems even if they don't
meet this definition. Here we use ``1e10`` as a somewhat arbitrary condition
number threshold to indicate a problem in our system.

.. doctest::
   :skipif: not scipy_available or not asl_available or not networkx_available

   >>> # PyomoNLP requires exactly one objective function
   >>> m._obj = pyo.Objective(expr=0.0)
   >>> nlp = PyomoNLP(m)
   >>> cond_threshold = 1e10
   >>> cond = np.linalg.cond(nlp.evaluate_jacobian_eq().toarray())
   >>> print(cond > cond_threshold)
   True

The system is poorly conditioned. Now we can check diagonal blocks of a block
triangularization to determine which blocks are causing the poor conditioning.

.. code-block:: python

   >>> var_blocks, con_blocks = igraph.block_triangularize()
   >>> for i, (vblock, cblock) in enumerate(zip(var_blocks, con_blocks)):
   ...     submatrix = nlp.extract_submatrix_jacobian(vblock, cblock)
   ...     cond = np.linalg.cond(submatrix.toarray())
   ...     print(f"block {i}: {cond}")
   ...     if cond > cond_threshold:
   ...         for var in vblock:
   ...             print(f"  {var.name}")
   ...         for con in cblock:
   ...             print(f"  {con.name}")
   block 0: 24.492504515710433
   block 1: 1.2480741394486336e+17
     flow
     flow_comp[1]
     flow_comp[2]
     flow_comp[3]
     sum_flow_eqn
     flow_eqn[1]
     flow_eqn[2]
     flow_eqn[3]

We see that the second block is causing the singularity, and that this block
contains the sum equation that we modified for this example. This suggests that
converting this equation to sum over flow rates rather than mass fractions just
converted a structural singularity to a numeric singularity, and didn't really
solve our problem. To see a fix that *does* resolve the singularity, see
:ref:`incidence_tutorial_dm`.
