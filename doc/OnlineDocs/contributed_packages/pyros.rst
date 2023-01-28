############
PyROS Solver
############

PyROS (Pyomo Robust Optimization Solver) is a Pyomo-based metasolver
for non-convex, two-stage adjustable robust optimization problems.

It was developed by **Natalie M. Isenberg**, **Jason A. F. Sherman**,
and **Chrysanthos E. Gounaris** of Carnegie Mellon University,
in collaboration with **John D. Siirola** of Sandia National Labs.
The developers gratefully acknowledge support from the U.S. Department of Energy's
`Institute for the Design of Advanced Energy Systems (IDAES) <https://idaes.org>`_.

Methodology Overview
-----------------------------

Below is an overview of the type of optimization models PyROS can accomodate.


* PyROS is suitable for optimization models of **continuous variables** that may feature non-linearities (including **non-convexities**) in both the variables and uncertain parameters.
* PyROS can handle **equality constraints** defining state variables, including implicit state variables that cannot be eliminated via reformulation.
* PyROS allows for **two-stage** optimization problems that may feature both first-stage and second-stage degrees of freedom.

PyROS is designed to operate on deterministic models of the general form

.. math::
    \begin{align*}
    \displaystyle \min_{\substack{x \in \mathcal{X}, \\ z \in \mathbb{R}^{n_z}, y\in\mathbb{R}^{n_y}}} & ~~ f_1\left(x\right) + f_2\left(x,z,y; q^{\text{nom}}\right) & \\
    \displaystyle \text{s.t.} \quad \: & ~~ g_i\left(x, z, y; q^{\text{nom}}\right) \leq 0 & \forall i \in \mathcal{I} \\
    & ~~ h_j\left(x,z,y; q^{\text{nom}}\right) = 0 & \forall j \in \mathcal{J} \\
    \end{align*}

where:

* :math:`x \in \mathcal{X}` are the "design" variables
  (i.e., first-stage degrees of freedom),
  where :math:`\mathcal{X} \subseteq \mathbb{R}^{n_x}` is the feasible space defined by the model constraints
  (including variable bounds specifications) referencing :math:`x` only.
* :math:`z \in \mathbb{R}^{n_z}` are the "control" variables
  (i.e., second-stage degrees of freedom)
* :math:`y \in \mathbb{R}^{n_y}` are the "state" variables
* :math:`q \in \mathbb{R}^{n_q}` is the vector of model parameters considered
  uncertain, and :math:`q^{\text{nom}}` is the vector of nominal values
  associated with those.
* :math:`f_1\left(x\right)` are the terms of the objective function that depend
  only on design variables
* :math:`f_2\left(x, z, y; q\right)` are the terms of the objective function
  that depend on all variables and the uncertain parameters
* :math:`g_i\left(x, z, y; q\right)` is the :math:`i^\text{th}`
  inequality constraint function in set :math:`\mathcal{I}` (see Note)
* :math:`h_j\left(x, z, y; q\right)` is the :math:`j^\text{th}`
  equality constraint function in set :math:`\mathcal{J}` (see Note)

.. note::
    PyROS accepts models in which bounds are directly imposed on
    ``Var`` objects representing components of the variables :math:`z`
    and :math:`y`. These models are cast to the form above
    by reformulating the bounds as inequality constraints.

.. note::
    A key requirement of PyROS is that each value of :math:`\left(x, z, q \right)`
    maps to a unique value of :math:`y`, a property that is assumed to
    be properly enforced by the system of equality constraints
    :math:`\mathcal{J}`.
    If the mapping is not unique, then the selection of 'state'
    (i.e., not degree of freedom) variables :math:`y` is incorrect,
    and one or more of the :math:`y` variables should be appropriately
    redesignated to be part of either :math:`x` or :math:`z`.

In order to cast the robust optimization counterpart formulation of the above model,
we now assume that the uncertain parameters may attain
any realization in a compact uncertainty set
:math:`\mathcal{Q} \subseteq \mathbb{R}^{n_q}` containing
the nominal value :math:`q^{\text{nom}}`.
The set :math:`\mathcal{Q}` may be **either continuous or discrete**.

Based on the above notation, the form of the robust counterpart addressed by PyROS is

.. math::
    \begin{align*}
    \displaystyle \min_{x \in \mathcal{X}}
    & \displaystyle \max_{q \in \mathcal{Q}}
    & \displaystyle \min_{z \in \mathbb{R}^{n_z}, y \in \mathbb{R}^{n_y}} \ \ & \displaystyle ~~ f_1\left(x\right) + f_2\left(x, z, y, q\right) & & \\
    & & \text{s.t.} \quad \:& \displaystyle ~~ g_i\left(x, z, y, q\right) \leq 0 &  & \forall i \in \mathcal{I}\\
    & & & \displaystyle ~~ h_j\left(x, z, y, q\right) = 0 &  & \forall j \in \mathcal{J}
    \end{align*}

PyROS solves problems of this form using the
Generalized Robust Cutting-Set algorithm developed in [Isenberg_et_al]_.

When using PyROS, please consider citing the above paper.

PyROS Required Inputs
-----------------------------
The required inputs to the PyROS solver are:

* The deterministic optimization model
* List of first-stage ("design") variables
* List of second-stage ("control") variables
* List of parameters considered uncertain
* The uncertainty set
* Subordinate local and global nonlinear programming (NLP) solvers

These are more elaborately presented in the next section.

.. note::
    Any variables in the model not specified to be first-stage or second-stage
    variables are automatically considered to be state variables.

PyROS Solver Interface
-----------------------------

.. autoclass:: pyomo.contrib.pyros.PyROS
    :members: solve

.. note::
    Upon successful convergence of PyROS, the solution returned is
    certified to be robust optimal only if
    (1) master problems are solved to global optimality (specified through
    the option ``solve_master_globally=True``);
    (2) a worst-case objective focus is chosen (by specifying the option
    ``objective_focus=ObjectiveType.worst_case``).

    Otherwise, the solution returned is certified to only be robust feasible.

PyROS Uncertainty Sets
-----------------------------
Uncertainty sets are represented by subclasses of
the :class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet` abstract base
class.
PyROS provides a suite of pre-implemented subclasses representing commonly used
uncertainty sets, including an
:class:`~pyomo.contrib.pyros.uncertainty_sets.IntersectionSet` subclass for
the intersection of a sequence of uncertainty set instances.
Custom user-defined uncertainty set types may be implemented by
subclassing the :class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`
class.

:ref:`This table <table-uncertsets>` provides mathematical definitions of
the various abstract and pre-implemented
:class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet` subclasses.

.. _table-uncertsets:

.. list-table:: Mathematical definitions of PyROS uncertainty sets of dimension :math:`n`.
   :header-rows: 1
   :class: tight-table

   * - Uncertainty Set Type
     - Input Data
     - Mathematical Definition
   * - ``BoxSet``
     - :math:`\begin{array}{l} q ^{\text{L}} \in \mathbb{R}^{n}, \\ q^{\text{U}} \in \mathbb{R}^{n} \end{array}`
     - :math:`\{q \in \mathbb{R}^n \mid q^\mathrm{L} \leq q \leq q^\mathrm{U}\}`
   * - ``CardinalitySet``
     - :math:`\begin{array}{l} q^{0} \in \mathbb{R}^{n}, \\ \hat{q} \in \mathbb{R}_{+}^{n}, \\ \Gamma \in [0, n] \end{array}`
     - :math:`\left\{ q \in \mathbb{R}^{n} \middle| \begin{array}{l} q = q^{0} + \hat{q} \circ \xi \\ \displaystyle \sum_{i=1}^{n} \xi_{i} \leq \Gamma \\ \xi \in [0, 1]^{n} \end{array} \right\}`
   * - ``BudgetSet``
     - :math:`\begin{array}{l} q^{0} \in \mathbb{R}^{n}, \\ b \in \mathbb{R}_{+}^{L}, \\ B \in \{0, 1\}^{L \times n} \end{array}`
     - :math:`\left\{ q \in \mathbb{R}^{n} \middle| \begin{array}{l} \begin{pmatrix} B \\ -I \end{pmatrix} q \leq \begin{pmatrix}  b + Bq^{0} \\ -q^{0} \end{pmatrix}  \end{array} \right\}`
   * - ``FactorModelSet``
     - :math:`\begin{array}{l} q^{0} \in \mathbb{R}^{n}, \\ \Psi \in \mathbb{R}^{n \times F}, \\ \beta \in [0, 1] \end{array}`
     - :math:`\left\{ q \in \mathbb{R}^{n} \middle| \begin{array}{l} q = q^{0} + \Psi \xi \\ \displaystyle\bigg| \sum_{j=1}^{F} \xi_{j} \bigg| \leq \beta F \\ \xi \in [-1, 1]^{F} \\ \end{array} \right\}`
   * - ``PolyhedralSet``
     - :math:`\begin{array}{l} A \in \mathbb{R}^{m \times n}, \\ b \in \mathbb{R}^{m}\end{array}`
     - :math:`\{q \in \mathbb{R}^{n} \mid A q \leq b\}`
   * - ``AxisAlignedEllipsoidalSet``
     - :math:`\begin{array}{l} q^0 \in \mathbb{R}^{n}, \\ \alpha \in \mathbb{R}_{+}^{n} \end{array}`
     - :math:`\left\{ q \in \mathbb{R}^{n} \middle| \begin{array}{l} \displaystyle\sum_{\substack{i = 1: \\ \alpha_{i} > 0}}^{n}  \left(\frac{q_{i} - q_{i}^{0}}{\alpha_{i}}\right)^2 \leq 1 \\ q_{i} = q_{i}^{0} \,\forall\,i : \alpha_{i} = 0 \end{array} \right\}`
   * - ``EllipsoidalSet``
     - :math:`\begin{array}{l} q^0 \in \mathbb{R}^n, \\ P \in \mathbb{S}_{++}^{n}, \\ s \in \mathbb{R}_{+} \end{array}`
     - :math:`\{q \in \mathbb{R}^{n} \mid (q - q^{0})^{\intercal} P^{-1} (q - q^{0}) \leq s\}`
   * - ``UncertaintySet``
     - :math:`g: \mathbb{R}^{n} \to \mathbb{R}^{m}`
     - :math:`\{q \in \mathbb{R}^{n} \mid g(q) \leq 0\}`
   * - ``DiscreteScenariosSet``
     - :math:`q^{1}, q^{2},\dots , q^{S} \in \mathbb{R}^{n}`
     - :math:`\{q^{1}, q^{2}, \dots , q^{S}\}`
   * - ``IntersectionSet``
     - :math:`\mathcal{Q}_{1}, \mathcal{Q}_{2}, \dots , \mathcal{Q}_{m} \subset \mathbb{R}^{n}`
     - :math:`\displaystyle \bigcap_{i=1}^{m} \mathcal{Q}_{i}`

.. note::
    Each of the PyROS uncertainty set classes inherits from the
    :class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`
    abstract base class.

PyROS Uncertainty Set Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.BoxSet
    :special-members: bounds, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.CardinalitySet
    :special-members: origin, positive_deviation, gamma, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.BudgetSet
    :special-members: coefficients_mat, rhs_vec, origin, budget_membership_mat, budget_rhs_vec, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.FactorModelSet
    :special-members: origin, number_of_factors, psi_mat, beta, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.PolyhedralSet
    :special-members: coefficients_mat, rhs_vec, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.AxisAlignedEllipsoidalSet
    :special-members: center, half_lengths, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.EllipsoidalSet
    :special-members: center, shape_matrix, scale, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.UncertaintySet
    :special-members: parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.DiscreteScenarioSet
    :special-members: scenarios, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.IntersectionSet
    :special-members: all_sets, type, parameter_bounds, dim, point_in_set


PyROS Usage Example
-----------------------------

We will use an example to illustrate the usage of PyROS. The problem we will use is called *hydro* and comes from the GAMS example problem database in `The GAMS Model Library <https://www.gams.com/latest/gamslib_ml/libhtml/>`_. The model was converted to Pyomo format via the `GAMS Convert tool <https://www.gams.com/latest/docs/S_CONVERT.html>`_.

This model is a QCQP with 31 variables. Of these variables, 13 represent degrees of freedom, with the additional 18 being state variables.
The model features 6 linear inequality constraints, 6 linear equality constraints, 6 non-linear (quadratic) equalities, and a quadratic objective.
We have augmented this model by converting one objective coefficient, two constraint coefficients, and one constraint right-hand side into Param objects so that they can be considered uncertain later on.

.. note::
    Per our analysis, the *hydro* problem satisfies the requirement that each value of :math:`\left(x, z, q \right)` maps to a unique value of :math:`y`, which indicates a proper partition of variables between (first- or second-stage) degrees of freedom and state variables.

Step 0: Import Pyomo and the PyROS Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In anticipation of using the PyROS solver and building the deterministic Pyomo model:

.. doctest::

  >>> # === Required import ===
  >>> import pyomo.environ as pyo
  >>> import pyomo.contrib.pyros as pyros

  >>> # === Instantiate the PyROS solver object ===
  >>> pyros_solver = pyo.SolverFactory("pyros")

Step 1: Define the Deterministic Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The deterministic Pyomo model for *hydro* is shown below.

.. note::
    Primitive data (Python literals) that have been hard-coded within a deterministic model cannot be later considered uncertain, unless they are first converted to ``Param`` objects within the ``ConcreteModel`` object.
    Furthermore, any ``Param`` object that is to be later considered uncertain must have the property ``mutable=True``.

.. note::
    In case modifying the ``mutable`` property inside the deterministic model object itself is not straight-forward in your context,
    you may consider adding the following statement **after** ``import pyomo.environ as pyo`` but **before** defining the model object:
    ``pyo.Param.DefaultMutable = True``. Note how this sets the default ``mutable`` property in all ``Param`` objects in the ensuing model instance to ``True``;
    consequently, this solution will not work with ``Param`` objects for which the ``mutable=False`` property was explicitly enabled inside the model object.

.. doctest::


  >>> # === Construct the Pyomo model object ===
  >>> m = pyo.ConcreteModel()
  >>> m.name = "hydro"

  >>> # === Define variables ===
  >>> m.x1 = pyo.Var(within=pyo.Reals,bounds=(150,1500),initialize=150)
  >>> m.x2 = pyo.Var(within=pyo.Reals,bounds=(150,1500),initialize=150)
  >>> m.x3 = pyo.Var(within=pyo.Reals,bounds=(150,1500),initialize=150)
  >>> m.x4 = pyo.Var(within=pyo.Reals,bounds=(150,1500),initialize=150)
  >>> m.x5 = pyo.Var(within=pyo.Reals,bounds=(150,1500),initialize=150)
  >>> m.x6 = pyo.Var(within=pyo.Reals,bounds=(150,1500),initialize=150)
  >>> m.x7 = pyo.Var(within=pyo.Reals,bounds=(0,1000),initialize=0)
  >>> m.x8 = pyo.Var(within=pyo.Reals,bounds=(0,1000),initialize=0)
  >>> m.x9 = pyo.Var(within=pyo.Reals,bounds=(0,1000),initialize=0)
  >>> m.x10 = pyo.Var(within=pyo.Reals,bounds=(0,1000),initialize=0)
  >>> m.x11 = pyo.Var(within=pyo.Reals,bounds=(0,1000),initialize=0)
  >>> m.x12 = pyo.Var(within=pyo.Reals,bounds=(0,1000),initialize=0)
  >>> m.x13 = pyo.Var(within=pyo.Reals,bounds=(0,None),initialize=0)
  >>> m.x14 = pyo.Var(within=pyo.Reals,bounds=(0,None),initialize=0)
  >>> m.x15 = pyo.Var(within=pyo.Reals,bounds=(0,None),initialize=0)
  >>> m.x16 = pyo.Var(within=pyo.Reals,bounds=(0,None),initialize=0)
  >>> m.x17 = pyo.Var(within=pyo.Reals,bounds=(0,None),initialize=0)
  >>> m.x18 = pyo.Var(within=pyo.Reals,bounds=(0,None),initialize=0)
  >>> m.x19 = pyo.Var(within=pyo.Reals,bounds=(0,None),initialize=0)
  >>> m.x20 = pyo.Var(within=pyo.Reals,bounds=(0,None),initialize=0)
  >>> m.x21 = pyo.Var(within=pyo.Reals,bounds=(0,None),initialize=0)
  >>> m.x22 = pyo.Var(within=pyo.Reals,bounds=(0,None),initialize=0)
  >>> m.x23 = pyo.Var(within=pyo.Reals,bounds=(0,None),initialize=0)
  >>> m.x24 = pyo.Var(within=pyo.Reals,bounds=(0,None),initialize=0)
  >>> m.x25 = pyo.Var(within=pyo.Reals,bounds=(100000,100000),initialize=100000)
  >>> m.x26 = pyo.Var(within=pyo.Reals,bounds=(60000,120000),initialize=60000)
  >>> m.x27 = pyo.Var(within=pyo.Reals,bounds=(60000,120000),initialize=60000)
  >>> m.x28 = pyo.Var(within=pyo.Reals,bounds=(60000,120000),initialize=60000)
  >>> m.x29 = pyo.Var(within=pyo.Reals,bounds=(60000,120000),initialize=60000)
  >>> m.x30 = pyo.Var(within=pyo.Reals,bounds=(60000,120000),initialize=60000)
  >>> m.x31 = pyo.Var(within=pyo.Reals,bounds=(60000,120000),initialize=60000)

  >>> # === Define parameters ===
  >>> m.set_of_params = pyo.Set(initialize=[0, 1, 2, 3])
  >>> nominal_values = {0:82.8*0.0016, 1:4.97, 2:4.97, 3:1800}
  >>> m.p = pyo.Param(m.set_of_params, initialize=nominal_values, mutable=True)

  >>> # === Specify the objective function ===
  >>> m.obj = pyo.Objective(expr=m.p[0]*m.x1**2 + 82.8*8*m.x1 + 82.8*0.0016*m.x2**2 +
  ...                               82.8*82.8*8*m.x2 + 82.8*0.0016*m.x3**2 + 82.8*8*m.x3 +
  ...                               82.8*0.0016*m.x4**2 + 82.8*8*m.x4 + 82.8*0.0016*m.x5**2 +
  ...                               82.8*8*m.x5 + 82.8*0.0016*m.x6**2 + 82.8*8*m.x6 + 248400,
  ...                               sense=pyo.minimize)

  >>> # === Specify the constraints ===
  >>> m.c2 = pyo.Constraint(expr=-m.x1 - m.x7 + m.x13 + 1200<= 0)
  >>> m.c3 = pyo.Constraint(expr=-m.x2 - m.x8 + m.x14 + 1500 <= 0)
  >>> m.c4 = pyo.Constraint(expr=-m.x3 - m.x9 + m.x15 + 1100 <= 0)
  >>> m.c5 = pyo.Constraint(expr=-m.x4 - m.x10 + m.x16 + m.p[3] <= 0)
  >>> m.c6 = pyo.Constraint(expr=-m.x5 - m.x11 + m.x17 + 950 <= 0)
  >>> m.c7 = pyo.Constraint(expr=-m.x6 - m.x12 + m.x18 + 1300 <= 0)
  >>> m.c8 = pyo.Constraint(expr=12*m.x19 - m.x25 + m.x26 == 24000)
  >>> m.c9 = pyo.Constraint(expr=12*m.x20 - m.x26 + m.x27 == 24000)
  >>> m.c10 = pyo.Constraint(expr=12*m.x21 - m.x27 + m.x28 == 24000)
  >>> m.c11 = pyo.Constraint(expr=12*m.x22 - m.x28 + m.x29 == 24000)
  >>> m.c12 = pyo.Constraint(expr=12*m.x23 - m.x29 + m.x30 == 24000)
  >>> m.c13 = pyo.Constraint(expr=12*m.x24 - m.x30 + m.x31 == 24000)
  >>> m.c14 = pyo.Constraint(expr=-8e-5*m.x7**2 + m.x13 == 0)
  >>> m.c15 = pyo.Constraint(expr=-8e-5*m.x8**2 + m.x14 == 0)
  >>> m.c16 = pyo.Constraint(expr=-8e-5*m.x9**2 + m.x15 == 0)
  >>> m.c17 = pyo.Constraint(expr=-8e-5*m.x10**2 + m.x16 == 0)
  >>> m.c18 = pyo.Constraint(expr=-8e-5*m.x11**2 + m.x17 == 0)
  >>> m.c19 = pyo.Constraint(expr=-8e-5*m.x12**2 + m.x18 == 0)
  >>> m.c20 = pyo.Constraint(expr=-4.97*m.x7 + m.x19 == 330)
  >>> m.c21 = pyo.Constraint(expr=-m.p[1]*m.x8 + m.x20 == 330)
  >>> m.c22 = pyo.Constraint(expr=-4.97*m.x9 + m.x21 == 330)
  >>> m.c23 = pyo.Constraint(expr=-4.97*m.x10 + m.x22 == 330)
  >>> m.c24 = pyo.Constraint(expr=-m.p[2]*m.x11 + m.x23 == 330)
  >>> m.c25 = pyo.Constraint(expr=-4.97*m.x12 + m.x24 == 330)

Step 2: Define the Uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, we need to collect into a list those ``Param`` objects of our model that represent potentially uncertain parameters. For purposes of our example, we shall assume uncertainty in the model parameters ``(m.p[0], m.p[1], m.p[2], m.p[3])``, for which we can conveniently utilize the ``m.p`` object (itself an indexed ``Param`` object).

.. doctest::

  >>> # === Specify which parameters are uncertain ===
  >>> uncertain_parameters = [m.p] # We can pass IndexedParams this way to PyROS, or as an expanded list per index

.. note::
    Any ``Param`` object that is to be considered uncertain by PyROS must have the property ``mutable=True``.

PyROS will seek to identify solutions that remain feasible for any realization of these parameters included in an uncertainty set. To that end, we need to construct an ``UncertaintySet`` object. In our example, let us utilize the ``BoxSet`` constructor to specify an uncertainty set of simple hyper-rectangular geometry. For this, we will assume each parameter value is uncertain within a percentage of its nominal value. Constructing this specific ``UncertaintySet`` object can be done as follows.

.. doctest::

  >>> # === Define the pertinent data ===
  >>> relative_deviation = 0.15
  >>> bounds = [(nominal_values[i] - relative_deviation*nominal_values[i],
  ...               nominal_values[i] + relative_deviation*nominal_values[i])
  ...               for i in range(4)]

  >>> # === Construct the desirable uncertainty set ===
  >>> box_uncertainty_set = pyros.BoxSet(bounds=bounds)

Step 3: Solve with PyROS
^^^^^^^^^^^^^^^^^^^^^^^^^^

PyROS requires the user to supply one local and one global NLP solver to be used for solving sub-problems. For convenience, we shall have PyROS invoke BARON as both the local and the global NLP solver.

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> # === Designate local and global NLP solvers ===
  >>> local_solver = pyo.SolverFactory('baron')
  >>> global_solver = pyo.SolverFactory('baron')

.. note::
    Additional solvers to be used as backup can be designated during the ``solve`` statement via the config options ``backup_local_solvers`` and ``backup_global_solvers`` presented above.

The final step in solving a model with PyROS is to designate the remaining required inputs, namely ``first_stage_variables`` and ``second_stage_variables``. Below, we present two separate cases.

PyROS Termination Conditions
"""""""""""""""""""""""""""""

PyROS will return one of six termination conditions upon completion. These termination conditions are tabulated below.

.. tabularcolumns:: |c|c|c|

+---------------------------------------------------+----------------------------------------------------------------+
|  **Termination Condition**                        | **Description**                                                |
+---------------------------------------------------+----------------------------------------------------------------+
|   ``pyrosTerminationCondition.robust_optimal``    |  The final solution is robust optimal                          |
+---------------------------------------------------+----------------------------------------------------------------+
|   ``pyrosTerminationCondition.robust_feasible``   |  The final solution is robust feasible                         |
+---------------------------------------------------+----------------------------------------------------------------+
|   ``pyrosTerminationCondition.robust_infeasible`` |  The posed problem is robust infeasible                        |
+---------------------------------------------------+----------------------------------------------------------------+
|   ``pyrosTerminationCondition.max_iter``          |  Maximum number of GRCS iteration reached                      |
+---------------------------------------------------+----------------------------------------------------------------+
|   ``pyrosTerminationCondition.time_out``          |  Maximum number of time reached                                |
+---------------------------------------------------+----------------------------------------------------------------+
|   ``pyrosTerminationCondition.subsolver_error``   |  Unacceptable return status(es) from a user-supplied sub-solver|
+---------------------------------------------------+----------------------------------------------------------------+


A Single-Stage Problem
"""""""""""""""""""""""""
If we choose to designate all variables as either design or state variables, without any control variables (i.e., all degrees of freedom are first-stage), we can use PyROS to solve the single-stage problem as shown below. In particular, let us instruct PyROS that variables ``m.x1`` through ``m.x6``, ``m.x19`` through ``m.x24``, and ``m.x31`` correspond to first-stage degrees of freedom.

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> # === Designate which variables correspond to first- and second-stage degrees of freedom ===
  >>> first_stage_variables =[m.x1, m.x2, m.x3, m.x4, m.x5, m.x6,
  ...                      m.x19, m.x20, m.x21, m.x22, m.x23, m.x24, m.x31]
  >>> second_stage_variables = []
  >>> # The remaining variables are implicitly designated to be state variables

  >>> # === Call PyROS to solve the robust optimization problem ===
  >>> results_1 = pyros_solver.solve(model = m,
  ...                                  first_stage_variables = first_stage_variables,
  ...                                  second_stage_variables = second_stage_variables,
  ...                                  uncertain_params = uncertain_parameters,
  ...                                  uncertainty_set = box_uncertainty_set,
  ...                                  local_solver = local_solver,
  ...                                  global_solver= global_solver,
  ...                                  options = {
  ...                                     "objective_focus": pyros.ObjectiveType.worst_case,
  ...                                     "solve_master_globally": True,
  ...                                     "load_solution":False
  ...                                   })
  ===========================================================================================
  PyROS: Pyomo Robust Optimization Solver ...
  ===========================================================================================
  ...
  INFO: Robust optimal solution identified. Exiting PyROS.

  >>> # === Query results ===
  >>> time = results_1.time
  >>> iterations = results_1.iterations
  >>> termination_condition = results_1.pyros_termination_condition
  >>> objective = results_1.final_objective_value
  >>> # === Print some results ===
  >>> single_stage_final_objective = round(objective,-1)
  >>> print("Final objective value: %s" % single_stage_final_objective)
  Final objective value: 48367380.0
  >>> print("PyROS termination condition: %s" % termination_condition)
  PyROS termination condition: pyrosTerminationCondition.robust_optimal

PyROS Results Object
"""""""""""""""""""""""""""
The results object returned by PyROS allows you to query the following information from the solve call:
total iterations of the algorithm ``iterations``, CPU time ``time``, the GRCS algorithm termination condition ``pyros_termination_condition``,
and the final objective function value ``final_objective_value``. If the option ``load_solution`` = ``True`` (default), the variables in the model will be
loaded to the solution determined by PyROS and can be obtained by querying the model variables. Note that in the results obtained above, we set ``load_solution`` = ``False``.
This is to ensure that the next set of runs shown here can utilize the original deterministic model, as the initial point can affect the performance of sub-solvers.

.. note::
    The reported ``final_objective_value`` and final model variable values depend on the selection of the option ``objective_focus``.
    The ``final_objective_value`` is the sum of first-stage and second-stage objective functions.
    If ``objective_focus = ObjectiveType.nominal``, second-stage objective and variables are evaluated at the nominal realization of the uncertain parameters, :math:`q^0`.
    If ``objective_focus = ObjectiveType.worst_case``, second-stage objective and variables are evaluated at the worst-case realization of the uncertain parameters, :math:`q^{k^\ast}` where :math:`k^\ast = argmax_{k \in \mathcal{K}} f_2(x,z^k,y^k,q^k)` .

An example of how to query these values on the previously obtained results is shown in the code above.



A Two-Stage Problem
""""""""""""""""""""""
For this next set of runs, we will assume that some of the previously designated first-stage degrees of freedom are in fact second-stage ones. PyROS handles second-stage degrees of freedom via the use of decision rules, which is controlled with the config option ``decision_rule_order`` presented above. Here, we shall select affine decision rules by setting ``decision_rule_order`` to the value of `1`.

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> # === Define the variable partitioning
  >>> first_stage_variables =[m.x5, m.x6, m.x19, m.x22, m.x23, m.x24, m.x31]
  >>> second_stage_variables = [m.x1, m.x2, m.x3, m.x4, m.x20, m.x21]
  >>> # The remaining variables are implicitly designated to be state variables

  >>> # === Call PyROS to solve the robust optimization problem ===
  >>> results_2 = pyros_solver.solve(model = m,
  ...                                   first_stage_variables = first_stage_variables,
  ...                                   second_stage_variables = second_stage_variables,
  ...                                   uncertain_params = uncertain_parameters,
  ...                                   uncertainty_set = box_uncertainty_set,
  ...                                   local_solver = local_solver,
  ...                                   global_solver = global_solver,
  ...                                   options = {
  ...                                      "objective_focus": pyros.ObjectiveType.worst_case,
  ...                                      "solve_master_globally": True,
  ...                                      "decision_rule_order": 1
  ...                                   })
  ===========================================================================================
  PyROS: Pyomo Robust Optimization Solver ...
  ...
  INFO: Robust optimal solution identified. Exiting PyROS.

  >>> # === Compare final objective to the singe-stage solution
  >>> two_stage_final_objective = round(pyo.value(results_2.final_objective_value),-1)
  >>> percent_difference = 100 * (two_stage_final_objective - single_stage_final_objective)/(single_stage_final_objective)
  >>> print("Percent objective change relative to constant decision rules objective: %.2f %%" % percent_difference)
  Percent objective change relative to constant decision rules objective: -24...

In this example, when we compare the final objective value in the case of constant decision rules (no second-stage recourse)
and affine decision rules, we see there is a ~25% decrease in total objective value.

The Price of Robustness
""""""""""""""""""""""""
Using appropriately constructed hierarchies, PyROS allows for the facile comparison of robust optimal objectives across sets to determine the "price of robustness."
For the set we considered here, the ``BoxSet``, we can create such a hierarchy via an array of ``relative_deviation`` parameters to define the size of these uncertainty sets.
We can then loop through this array and call PyROS within a loop to identify robust solutions in light of each of the specified ``BoxSet`` objects.

.. code::

  >>> # This takes a long time to run and therefore is not a doctest
  >>> # === An array of maximum relative deviations from the nominal uncertain parameter values to utilize in constructing box sets
  >>> relative_deviation_list = [0.00, 0.10, 0.20, 0.30, 0.40]
  >>> # === Final robust optimal objectives
  >>> robust_optimal_objectives = []
  >>> for relative_deviation in relative_deviation_list: # doctest: +SKIP
  ...   bounds = [(nominal_values[i] - relative_deviation*nominal_values[i],
  ...                   nominal_values[i] + relative_deviation*nominal_values[i])
  ...                   for i in range(4)]
  ...   box_uncertainty_set = pyros.BoxSet(bounds = bounds)
  ...   results = pyros_solver.solve(model = m,
  ...                                     first_stage_variables = first_stage_variables,
  ...                                     second_stage_variables = second_stage_variables,
  ...                                     uncertain_params = uncertain_parameters,
  ...                                     uncertainty_set = box_uncertainty_set,
  ...                                     local_solver = local_solver,
  ...                                     global_solver = global_solver,
  ...                                     options = {
  ...                                        "objective_focus": pyros.ObjectiveType.worst_case,
  ...                                        "solve_master_globally": True,
  ...                                        "decision_rule_order": 1
  ...                                     })
  ...   if results.pyros_termination_condition != pyros.pyrosTerminationCondition.robust_optimal:
  ...       print("This instance didn't solve to robust optimality.")
  ...       robust_optimal_objective.append("-----")
  ...   else:
  ...       robust_optimal_objectives.append(str(results.final_objective_value))

For this example, we obtain the following price of robustness results:

.. tabularcolumns:: |c|c|c|

+------------------------------------------+------------------------------+-----------------------------+
|  **Uncertainty Set Size (+/-)** :sup:`o` | **Robust Optimal Objective** | **% Increase** :sup:`x`     |
+------------------------------------------+------------------------------+-----------------------------+
|   0.00                                   | 35,837,659.18                | 0.00 %                      |
+------------------------------------------+------------------------------+-----------------------------+
|   0.10                                   | 36,135,191.59                | 0.82 %                      |
+------------------------------------------+------------------------------+-----------------------------+
|   0.20                                   | 36,437,979.81                | 1.64 %                      |
+------------------------------------------+------------------------------+-----------------------------+
|   0.30                                   | 43,478,190.92                | 17.57 %                     |
+------------------------------------------+------------------------------+-----------------------------+
|   0.40                                   | ``robust_infeasible``        | :math:`\text{-----}`        |
+------------------------------------------+------------------------------+-----------------------------+

Note how, in the case of the last uncertainty set, we were able to utilize PyROS to show the robust infeasibility of the problem.

:sup:`o` **Relative Deviation from Nominal Realization**

:sup:`x` **Relative to Deterministic Optimal Objective**

This clearly illustrates the impact that the uncertainty set size can have on the robust optimal objective values.
Price of robustness studies like this are easily implemented using PyROS.

.. warning::

    PyROS is still under a beta release. Please provide feedback and/or
    report any problems by opening an issue on the Pyomo
    `GitHub page <https://github.com/Pyomo/pyomo/issues/new/choose>`_.
