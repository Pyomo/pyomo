.. _analysis_nlp_initialization:

NLP Initialization
******************

.. warning::

   This package lives in :mod:`pyomo.devel`. APIs, options, and behavior may
   change without notice.

The initialization module within ``pyomo.devel.initialization`` is intended to 
provide methods to help initialize nonconvex nonlinear programs (NLPs). The 
goal is to increase the chance of finding a local minimizer (i.e., decrease the
chance of getting stuck a point that locally minimizes infeasibility). If 
you are already able to solve your problem with a local NLP solver, these 
tools will not help you. Example usage is shown below.

.. literalinclude:: /../../pyomo/devel/initialization/examples/init_polynomial_ex.py
    :start-after: # === Required imports ===

The :func:`initialize_nlp <pyomo.devel.initialization.initialize.initialize_nlp>` 
function uses the specified method to try to find a good starting point for the
NLP solver and then attempts to solve the problem with the given NLP solver. 

.. note::
    
    Currently, this module only works with solvers from :mod:`pyomo.contrib.solver`.


Initialization Methods
======================

The initialization method is selected using the 
:class:`InitializationMethod <pyomo.devel.initialization.initialize.InitializationMethod>` enum.

.. note::

    Not all of the methods described below require all nonlinear variables to be 
    bounded. However, all of the methods will perform better if all nonlinear 
    variables are bounded (the tighter the bounds, the better).


Method ``global_opt``
---------------------

This method uses an MINLP solver to try to find a feasible solution. We 
adjust the solver parameters so that the solver will stop as soon as any 
feasible solution is found. We then initialize the NLP solver at that 
feasible solution. Many MINLP solvers will default to a very large 
time limit, so it can be useful to specify a time limit before 
calling :func:`initialize_nlp <pyomo.devel.initialization.initialize.initialize_nlp>`:

.. testcode::
   :skipif: not scip_available

   import pyomo.environ as pyo
   from pyomo.contrib.solver.common.factory import SolverFactory

   global_solver = SolverFactory('scip_direct')
   global_solver.config.time_limit = 600  # 10 minutes
   # now call initialize_nlp

This method currently works with the following solver interfaces for MINLP solvers:

* SCIP (:class:`direct <pyomo.contrib.solver.solvers.scip.scip_direct.ScipDirect>` and
  :class:`persistent <pyomo.contrib.solver.solvers.scip.scip_direct.ScipPersistent>`)
* :class:`Gurobi MINLP <pyomo.contrib.solver.solvers.gurobi.gurobi_direct_minlp.GurobiDirectMINLP>`

Advantages
^^^^^^^^^^

* Currently, this is the method that is most likely to succeed in finding a 
  feasible solution. 
* Does not strictly require variable bounds

Disadvantages
^^^^^^^^^^^^^

* This method will only work if the model is completely algebraic. It will not
  work with external functions.


Method ``pwl_approximation``
----------------------------

This method builds a piecewise linear (PWL) approximation of the model, solves
it, and initializes the NLP solver at the solutin. If the NLP solver does not 
converge, then the PWL approximation will be refined by adding additional 
"segments". This is repeated until either a feasible solution is found or 
the iteration limit is reached. 

This method does not currently work as well as ``global_opt``, but it does
have a great deal of potential. We expect future versions of this method 
to perform significantly better. 

Advantages
^^^^^^^^^^

* Does not require an MINLP solver
* Future versions will work with external functions

Disadvantages
^^^^^^^^^^^^^

* Current implemenation can be slow
* Requires all nonlinear variables to be bounded


Method ``lp_approximation``
---------------------------

This method is similar to the PWL approximation method, but it builds
an LP approximation instead and does not do any refinement. Another 
distinction is that the LP approximation uses a linear least-squares 
fit, so the approximation may not equal the original function at the
variable bounds. This also means that variable bounds are not strictly
necessary, though they do help improve the approximation.

Advantages
^^^^^^^^^^

* Fast
* Future versions will work with external functions
* Does not strictly require variable bounds
* Does not require an MINLP or even an MILP solver

Disadvantages
^^^^^^^^^^^^^

* This method only attempts to initialize the problem once. If it does
  not succeed, it is done.
