####################################
Trust Region Framework Method Solver
####################################

The Trust Region Framework (TRF) method solver allows users to solve hybrid
glass box/black box optimization problems in which parts of the system are
modeled with open, equation-based models and parts of the system are black
boxes. This method utilizes surrogate models that substitute high-fidelity
models with low-fidelity basis functions, thus avoiding the direct implementation
of the large, computationally expensive high-fidelity models. This is done
iteratively, resulting in fewer calls to the computationally expensive functions.

This module implements the method from Yoshio & Biegler
[`Yoshio & Biegler, 2021`_] and represents a rewrite of the original 2018
implementation of the algorithm from Eason & Biegler [`Eason & Biegler, 2018`_].

In the context of this updated module, black box functions are implemented as
Pyomo External Functions.

This work was conducted as part of the Institute for the Design of Advanced
Energy Systems (`IDAES <https://idaes.org>`_) with support through the
Simulation-Based Engineering, Crosscutting Research Program within the U.S.
Department of Energy’s Office of Fossil Energy and Carbon Management.

.. _Eason & Biegler, 2018: https://doi.org/10.1002/aic.16364
.. _Yoshio & Biegler, 2021: https://doi.org/10.1002/aic.17054

Methodology Overview
---------------------

The formulation of the original hybrid problem is:

.. math::
   :nowrap:

   \[\begin{array}{ll}
    \min_{} & f\left(z, w, d\left(w\right)\right) \\
    \text{s.t.} & h\left(z, w, d\left(w\right)\right) = 0 \\
    & g\left(z, w, d\left(w\right)\right) \leq 0
   \end{array}\]

where:

* :math:`w \in \mathbb{R}^m` are the inputs to the external functions
* :math:`z \in \mathbb{R}^n` are the remaining decision variables (i.e., degrees of freedom)
* :math:`d(w) : \mathbb{R}^m \to \mathbb{R}^p` are the outputs of the external functions as a function of :math:`w`
* :math:`f`, `h`, `g`, `d` are all assumed to be twice continuously differentiable

This formulation is reworked to separate all external function information as
follows to enable the usage of the trust region method:

.. math::
   :nowrap:

   \[\begin{array}{ll}
     \min_{x} & f\left(x\right) \\
     \text{s.t.} & h\left(x\right) = 0 \\
     & g\left(x\right) \leq 0 \\
     & y = d\left(w\right)
   \end{array}\]

where:

* :math:`y \in \mathbb{R}^p` are the outputs of the external functions
* :math:`x^T = [w^T, y^T, z^T]` is a set of all inputs and outputs

Using this formulation and a user-supplied low-fidelity/ideal model basis function
:math:`b\left(w\right)`, the algorithm iteratively solves subproblems using
the surrogate model:

.. math::
    r_k\left(w\right) = b\left(w\right) + \left( d\left(w_k\right) - b\left(w_k\right) \right) + \left( \nabla d\left(w_k\right) - \nabla b\left(w_k\right) \right)^T \left( w - w_k \right)

This acts similarly to Newton's method in that small, incremental steps are taken
towards an optimal solution. At each iteration, the current solution of the
subproblem is compared to the previous solution to ensure that
the iteration has moved in a direction towards an optimal solution. If not true,
the step is rejected. If true, the step is accepted and the surrogate
model is updated for the next iteration.

When using TRF, please consider citing the above papers.

TRF Inputs
-----------

The required inputs to the TRF
:py:meth:`solve <pyomo.contrib.trustregion.TRF.TrustRegionSolver.solve>`
method are the following:

* The optimization model
* List of degree of freedom variables within the model

The optional input to the TRF
:py:meth:`solve <pyomo.contrib.trustregion.TRF.TrustRegionSolver.solve>`
method is the following:

* The external function surrogate model rule ("basis function")


TRF Solver Interface
---------------------

.. note::
    The keyword arguments can be updated at solver instantiation or later when the ``solve`` method is called.

.. autoclass:: pyomo.contrib.trustregion.TRF.TrustRegionSolver
    :noindex:
    :members: solve

TRF Usage Example
------------------
Two examples can be found in the examples_ subdirectory. One of them is
implemented below.

.. _examples: https://github.com/Pyomo/pyomo/tree/main/pyomo/contrib/trustregion/examples

Step 0: Import Pyomo
^^^^^^^^^^^^^^^^^^^^^

.. doctest::

   >>> # === Required imports ===
   >>> import pyomo.environ as pyo

Step 1: Define the external function and its gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doctest::

   >>> # === Define a 'black box' function and its gradient ===
   >>> def ext_fcn(a, b):
   ...     return pyo.sin(a - b)
   >>> def grad_ext_fcn(args, fixed):
   ...     a, b = args[:2]
   ...     return [ pyo.cos(a - b), -pyo.cos(a - b) ]

Step 2: Create the model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doctest::

   >>> # === Construct the Pyomo model object ===
   >>> def create_model():
   ...     m = pyo.ConcreteModel()
   ...     m.name = 'Example 1: Eason'
   ...     m.z = pyo.Var(range(3), domain=pyo.Reals, initialize=2.)
   ...     m.x = pyo.Var(range(2), initialize=2.)
   ...     m.x[1] = 1.0
   ...
   ...     m.ext_fcn = pyo.ExternalFunction(ext_fcn, grad_ext_fcn)
   ...
   ...     m.obj = pyo.Objective(
   ...         expr=(m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 \
   ...            + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6
   ...     )
   ...
   ...     m.c1 = pyo.Constraint(
   ...         expr=m.x[0] * m.z[0]**2 + m.ext_fcn(m.x[0], m.x[1]) == 2*pyo.sqrt(2.0)
   ...         )
   ...     m.c2 = pyo.Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+pyo.sqrt(2.0))
   ...     return m
   >>> model = create_model()

Step 3: Solve with TRF
^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    Reminder from earlier that the ``solve`` method requires the user pass the model and a list of variables
    which represent the degrees of freedom in the model. The user may also pass
    a low-fidelity/ideal model (or "basis function") to this method to improve
    convergence.

.. doctest::
   :skipif: not ipopt_available

   >>> # === Instantiate the TRF solver object ===
   >>> trf_solver = pyo.SolverFactory('trustregion')
   >>> # === Solve with TRF ===
   >>> result = trf_solver.solve(model, [model.z[0], model.z[1], model.z[2]])
   EXIT: Optimal solution found.
   ...

The :py:meth:`solve <pyomo.contrib.trustregion.TRF.TrustRegionSolver.solve>`
method returns a clone of the original model which has been run
through TRF algorithm, thus leaving the original model intact.


.. warning::

    TRF is still under a beta release. Please provide feedback and/or
    report any problems by opening an issue on the Pyomo
    `GitHub page <https://github.com/Pyomo/pyomo/issues/new/choose>`_.
