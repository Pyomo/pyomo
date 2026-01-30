.. _pyros_overview:

==========================
PyROS Methodology Overview
==========================

PyROS can accommodate optimization models with:

* **Continuous variables** only
* **Nonlinearities** (including **nonconvexities**) in both the
  variables and uncertain parameters
* **First-stage degrees of freedom** and **second-stage degrees of freedom**
* **Equality constraints** defining state variables,
  including implicitly defined state variables that cannot be
  eliminated from the model via reformulation
* **Uncertain parameters** participating in the inequality constraints,
  equality constraints, and/or objective function

Supported deterministic models are nonlinear programs (NLPs)
of the general form

.. _deterministic-model:

.. math::
   :nowrap:

   \[\begin{array}{clll}
    \displaystyle \min_{\substack{x \in \mathcal{X}, \\ z \in \mathbb{R}^{n_z}, \\ y\in\mathbb{R}^{n_y}}}
      & ~~ f_1\left(x\right) + f_2(x,z,y; q^{\text{nom}}) & \\
    \displaystyle \text{s.t.} & ~~ g_i(x, z, y; q^{\text{nom}}) \leq 0 & \forall\,i \in \mathcal{I} \\
    & ~~ h_j(x,z,y; q^{\text{nom}}) = 0 & \forall\,j \in \mathcal{J} \\
   \end{array}\]

where:

* :math:`x \in \mathcal{X}` denotes the first-stage degree of freedom variables
  (or design variables),
  of which the feasible space :math:`\mathcal{X} \subseteq \mathbb{R}^{n_x}`
  is defined by the model constraints
  (including variable bounds specifications) referencing :math:`x` only
* :math:`z \in \mathbb{R}^{n_z}` denotes the second-stage degree of freedom
  variables (or control variables)
* :math:`y \in \mathbb{R}^{n_y}` denotes the state variables
* :math:`q \in \mathbb{R}^{n_q}` denotes the model parameters considered
  uncertain, and :math:`q^{\text{nom}}` is the vector of nominal values
  associated with those
* :math:`f_1\left(x\right)` is the summand of the objective function that depends
  only on the first-stage degree of freedom variables
* :math:`f_2\left(x, z, y; q\right)` is the summand of the objective function
  that depends on all variables and the uncertain parameters
* :math:`g_i\left(x, z, y; q\right)` is the :math:`i^\text{th}`
  inequality constraint function in set :math:`\mathcal{I}`
  (see :ref:`first Note <var-bounds-to-ineqs>`)
* :math:`h_j\left(x, z, y; q\right)` is the :math:`j^\text{th}`
  equality constraint function in set :math:`\mathcal{J}`
  (see :ref:`second Note <pyros_unique_state_vars>`)

.. _var-bounds-to-ineqs:

.. note::

   PyROS accepts and automatically reformulates models with:

   1. Interval bounds on components of :math:`(x, z, y)`
   2. Ranged inequality constraints


.. _pyros_unique_state_vars:

.. note::
    A key assumption of PyROS is that
    for every
    :math:`x \in \mathcal{X}`,
    :math:`z \in \mathbb{R}^{n_z}`,
    :math:`q \in \mathcal{Q}`,
    there exists a unique :math:`y \in \mathbb{R}^{n_y}`
    for which :math:`(x, z, y, q)`
    satisfies the equality constraints
    :math:`h_j(x, z, y, q) = 0\,\,\forall\, j \in \mathcal{J}`.
    If this assumption is not met,
    then the selection of state
    (i.e., not degree of freedom) variables :math:`y` is incorrect,
    and one or more entries of :math:`y` should be appropriately
    redesignated to be part of either :math:`x` or :math:`z`.

In order to cast the robust optimization counterpart of the
:ref:`deterministic model <deterministic-model>`,
we now assume that the uncertain parameters :math:`q` may attain
any realization in a compact uncertainty set
:math:`\mathcal{Q} \subseteq \mathbb{R}^{n_q}` containing
the nominal value :math:`q^{\text{nom}}`.
The set :math:`\mathcal{Q}` may be **either continuous or discrete**.

Based on the above notation,
the form of the robust counterpart addressed by PyROS is

.. math::
   :nowrap:

   \[\begin{array}{ccclll}
    \displaystyle \min_{x \in \mathcal{X}}
    & \displaystyle \max_{q \in \mathcal{Q}}
    & \displaystyle \min_{\substack{z \in \mathbb{R}^{n_z},\\y \in \mathbb{R}^{n_y}}} \ \ & \displaystyle ~~ f_1\left(x\right) + f_2\left(x, z, y, q\right) \\
    & & \text{s.t.}~ & \displaystyle ~~ g_i\left(x, z, y, q\right) \leq 0 &  & \forall\, i \in \mathcal{I}\\
    & & & \displaystyle ~~ h_j\left(x, z, y, q\right) = 0 &  & \forall\,j \in \mathcal{J}
   \end{array}\]

PyROS accepts a deterministic model and accompanying uncertainty set
and then, using the Generalized Robust Cutting-Set algorithm developed
in [IAE+21]_, seeks a solution to the robust counterpart.
