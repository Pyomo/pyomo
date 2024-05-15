############
PyROS Solver
############

PyROS (Pyomo Robust Optimization Solver) is a Pyomo-based meta-solver
for non-convex, two-stage adjustable robust optimization problems.

It was developed by **Natalie M. Isenberg**, **Jason A. F. Sherman**,
and **Chrysanthos E. Gounaris** of Carnegie Mellon University,
in collaboration with **John D. Siirola** of Sandia National Labs.
The developers gratefully acknowledge support from the U.S. Department of Energy's
`Institute for the Design of Advanced Energy Systems (IDAES) <https://idaes.org>`_.

Methodology Overview
-----------------------------

Below is an overview of the type of optimization models PyROS can accommodate.


* PyROS is suitable for optimization models of **continuous variables**
  that may feature non-linearities (including **non-convexities**) in
  both the variables and uncertain parameters.
* PyROS can handle **equality constraints** defining state variables,
  including implicit state variables that cannot be eliminated via
  reformulation.
* PyROS allows for **two-stage** optimization problems that may
  feature both first-stage and second-stage degrees of freedom.

PyROS is designed to operate on deterministic models of the general form

.. _deterministic-model:

.. math::
    \begin{array}{clll}
    \displaystyle \min_{\substack{x \in \mathcal{X}, \\ z \in \mathbb{R}^{n_z}, y\in\mathbb{R}^{n_y}}} & ~~ f_1\left(x\right) + f_2(x,z,y; q^{\text{nom}}) & \\
    \displaystyle \text{s.t.} & ~~ g_i(x, z, y; q^{\text{nom}}) \leq 0 & \forall\,i \in \mathcal{I} \\
    & ~~ h_j(x,z,y; q^{\text{nom}}) = 0 & \forall\,j \in \mathcal{J} \\
    \end{array}

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
  inequality constraint function in set :math:`\mathcal{I}`
  (see :ref:`Note <var-bounds-to-ineqs>`)
* :math:`h_j\left(x, z, y; q\right)` is the :math:`j^\text{th}`
  equality constraint function in set :math:`\mathcal{J}`
  (see :ref:`Note <unique-mapping>`)

.. _var-bounds-to-ineqs:

.. note::
    PyROS accepts models in which bounds are directly imposed on
    ``Var`` objects representing components of the variables :math:`z`
    and :math:`y`. These models are cast to
    :ref:`the form above <deterministic-model>`
    by reformulating the bounds as inequality constraints.

.. _unique-mapping:

.. note::
    A key requirement of PyROS is that each value of :math:`\left(x, z, q \right)`
    maps to a unique value of :math:`y`, a property that is assumed to
    be properly enforced by the system of equality constraints
    :math:`\mathcal{J}`.
    If the mapping is not unique, then the selection of 'state'
    (i.e., not degree of freedom) variables :math:`y` is incorrect,
    and one or more of the :math:`y` variables should be appropriately
    redesignated to be part of either :math:`x` or :math:`z`.

In order to cast the robust optimization counterpart of the
:ref:`deterministic model <deterministic-model>`,
we now assume that the uncertain parameters may attain
any realization in a compact uncertainty set
:math:`\mathcal{Q} \subseteq \mathbb{R}^{n_q}` containing
the nominal value :math:`q^{\text{nom}}`.
The set :math:`\mathcal{Q}` may be **either continuous or discrete**.

Based on the above notation, the form of the robust counterpart addressed by PyROS is

.. math::
    \begin{array}{ccclll}
    \displaystyle \min_{x \in \mathcal{X}}
    & \displaystyle \max_{q \in \mathcal{Q}}
    & \displaystyle \min_{\substack{z \in \mathbb{R}^{n_z},\\y \in \mathbb{R}^{n_y}}} \ \ & \displaystyle ~~ f_1\left(x\right) + f_2\left(x, z, y, q\right) \\
    & & \text{s.t.}~ & \displaystyle ~~ g_i\left(x, z, y, q\right) \leq 0 &  & \forall\, i \in \mathcal{I}\\
    & & & \displaystyle ~~ h_j\left(x, z, y, q\right) = 0 &  & \forall\,j \in \mathcal{J}
    \end{array}

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

These are more elaborately presented in the
:ref:`Solver Interface <solver-interface>` section.

.. note::
    Any variables in the model not specified to be first-stage or second-stage
    variables are automatically considered to be state variables.

.. _solver-interface:

PyROS Solver Interface
-----------------------------

.. autoclass:: pyomo.contrib.pyros.PyROS
    :members: solve

.. note::
    Upon successful convergence of PyROS, the solution returned is
    certified to be robust optimal only if:

    1. master problems are solved to global optimality
       (by specifying ``solve_master_globally=True``)
    2. a worst-case objective focus is chosen
       (by specifying ``objective_focus=ObjectiveType.worst_case``)

    Otherwise, the solution returned is certified to only be robust feasible.


PyROS Uncertainty Sets
-----------------------------
Uncertainty sets are represented by subclasses of
the :class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`
abstract base class.
PyROS provides a suite of pre-implemented subclasses representing
commonly used uncertainty sets.
Custom user-defined uncertainty set types may be implemented by
subclassing the
:class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet` class.
The intersection of a sequence of concrete
:class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`
instances can be easily constructed by instantiating the pre-implemented
:class:`~pyomo.contrib.pyros.uncertainty_sets.IntersectionSet`
subclass.

The table that follows provides mathematical definitions of
the various abstract and pre-implemented
:class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet` subclasses.

.. _table-uncertsets:

.. list-table:: Mathematical definitions of PyROS uncertainty sets of dimension :math:`n`.
   :header-rows: 1
   :class: tight-table

   * - Uncertainty Set Type
     - Input Data
     - Mathematical Definition
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.BoxSet`
     - :math:`\begin{array}{l} q ^{\text{L}} \in \mathbb{R}^{n}, \\ q^{\text{U}} \in \mathbb{R}^{n} \end{array}`
     - :math:`\{q \in \mathbb{R}^n \mid q^\mathrm{L} \leq q \leq q^\mathrm{U}\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.CardinalitySet`
     - :math:`\begin{array}{l} q^{0} \in \mathbb{R}^{n}, \\ \hat{q} \in \mathbb{R}_{+}^{n}, \\ \Gamma \in [0, n] \end{array}`
     - :math:`\left\{ q \in \mathbb{R}^{n} \middle| \begin{array}{l} q = q^{0} + \hat{q} \circ \xi \\ \displaystyle \sum_{i=1}^{n} \xi_{i} \leq \Gamma \\ \xi \in [0, 1]^{n} \end{array} \right\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.BudgetSet`
     - :math:`\begin{array}{l} q^{0} \in \mathbb{R}^{n}, \\ b \in \mathbb{R}_{+}^{L}, \\ B \in \{0, 1\}^{L \times n} \end{array}`
     - :math:`\left\{ q \in \mathbb{R}^{n} \middle| \begin{array}{l} \begin{pmatrix} B \\ -I \end{pmatrix} q \leq \begin{pmatrix}  b + Bq^{0} \\ -q^{0} \end{pmatrix}  \end{array} \right\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.FactorModelSet`
     - :math:`\begin{array}{l} q^{0} \in \mathbb{R}^{n}, \\ \Psi \in \mathbb{R}^{n \times F}, \\ \beta \in [0, 1] \end{array}`
     - :math:`\left\{ q \in \mathbb{R}^{n} \middle| \begin{array}{l} q = q^{0} + \Psi \xi \\ \displaystyle\bigg| \sum_{j=1}^{F} \xi_{j} \bigg| \leq \beta F \\ \xi \in [-1, 1]^{F} \\ \end{array} \right\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.PolyhedralSet`
     - :math:`\begin{array}{l} A \in \mathbb{R}^{m \times n}, \\ b \in \mathbb{R}^{m}\end{array}`
     - :math:`\{q \in \mathbb{R}^{n} \mid A q \leq b\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.AxisAlignedEllipsoidalSet`
     - :math:`\begin{array}{l} q^0 \in \mathbb{R}^{n}, \\ \alpha \in \mathbb{R}_{+}^{n} \end{array}`
     - :math:`\left\{ q \in \mathbb{R}^{n} \middle| \begin{array}{l} \displaystyle\sum_{\substack{i = 1: \\ \alpha_{i} > 0}}^{n}  \left(\frac{q_{i} - q_{i}^{0}}{\alpha_{i}}\right)^2 \leq 1 \\ q_{i} = q_{i}^{0} \,\forall\,i : \alpha_{i} = 0 \end{array} \right\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.EllipsoidalSet`
     - :math:`\begin{array}{l} q^0 \in \mathbb{R}^n, \\ P \in \mathbb{S}_{++}^{n}, \\ s \in \mathbb{R}_{+} \end{array}`
     - :math:`\{q \in \mathbb{R}^{n} \mid (q - q^{0})^{\intercal} P^{-1} (q - q^{0}) \leq s\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`
     - :math:`g: \mathbb{R}^{n} \to \mathbb{R}^{m}`
     - :math:`\{q \in \mathbb{R}^{n} \mid g(q) \leq 0\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.DiscreteScenarioSet`
     - :math:`q^{1}, q^{2},\dots , q^{S} \in \mathbb{R}^{n}`
     - :math:`\{q^{1}, q^{2}, \dots , q^{S}\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.IntersectionSet`
     - :math:`\mathcal{Q}_{1}, \mathcal{Q}_{2}, \dots , \mathcal{Q}_{m} \subset \mathbb{R}^{n}`
     - :math:`\displaystyle \bigcap_{i=1}^{m} \mathcal{Q}_{i}`

.. note::
    Each of the PyROS uncertainty set classes inherits from the
    :class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`
    abstract base class.

PyROS Uncertainty Set Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.BoxSet
    :show-inheritance:
    :special-members: bounds, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.CardinalitySet
    :show-inheritance:
    :special-members: origin, positive_deviation, gamma, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.BudgetSet
    :show-inheritance:
    :special-members: coefficients_mat, rhs_vec, origin, budget_membership_mat, budget_rhs_vec, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.FactorModelSet
    :show-inheritance:
    :special-members: origin, number_of_factors, psi_mat, beta, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.PolyhedralSet
    :show-inheritance:
    :special-members: coefficients_mat, rhs_vec, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.AxisAlignedEllipsoidalSet
    :show-inheritance:
    :special-members: center, half_lengths, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.EllipsoidalSet
    :show-inheritance:
    :special-members: center, shape_matrix, scale, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.UncertaintySet
    :show-inheritance:
    :special-members: parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.DiscreteScenarioSet
    :show-inheritance:
    :special-members: scenarios, type, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.IntersectionSet
    :show-inheritance:
    :special-members: all_sets, type, parameter_bounds, dim, point_in_set


PyROS Usage Example
-----------------------------

In this section, we illustrate the usage of PyROS with a modeling example.
The deterministic problem of interest is called *hydro* 
(available `here <https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_hydro.html>`_),
a QCQP taken from the
`GAMS Model Library <https://www.gams.com/latest/gamslib_ml/libhtml/>`_.
We have converted the model to Pyomo format using the
`GAMS Convert tool <https://www.gams.com/latest/docs/S_CONVERT.html>`_.

The *hydro* model features 31 variables,
of which 13 are degrees of freedom and 18 are state variables.
Moreover, there are
6 linear inequality constraints,
12 linear equality constraints,
6 non-linear (quadratic) equality constraints,
and a quadratic objective.
We have extended this model by converting one objective coefficient,
two constraint coefficients, and one constraint right-hand side
into ``Param`` objects so that they can be considered uncertain later on.

.. note::
    Per our analysis, the *hydro* problem satisfies the requirement that
    each value of :math:`\left(x, z, q \right)` maps to a unique
    value of :math:`y`, which, in accordance with
    :ref:`our earlier note <unique-mapping>`,
    indicates a proper partitioning of the model variables
    into (first-stage and second-stage) degrees of freedom and
    state variables.

Step 0: Import Pyomo and the PyROS Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In anticipation of using the PyROS solver and building the deterministic Pyomo
model:

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
    Primitive data (Python literals) that have been hard-coded within a
    deterministic model cannot be later considered uncertain,
    unless they are first converted to ``Param`` objects within
    the ``ConcreteModel`` object.
    Furthermore, any ``Param`` object that is to be later considered
    uncertain must have the property ``mutable=True``.

.. note::
    In case modifying the ``mutable`` property inside the deterministic
    model object itself is not straightforward in your context,
    you may consider adding the following statement **after**
    ``import pyomo.environ as pyo`` but **before** defining the model
    object: ``pyo.Param.DefaultMutable = True``.
    For all ``Param`` objects declared after this statement,
    the attribute ``mutable`` is set to ``True`` by default.
    Hence, non-mutable ``Param`` objects are now declared by
    explicitly passing the argument ``mutable=False`` to the
    ``Param`` constructor.

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

First, we need to collect into a list those ``Param`` objects of our model
that represent potentially uncertain parameters.
For the purposes of our example, we shall assume uncertainty in the model
parameters ``[m.p[0], m.p[1], m.p[2], m.p[3]]``, for which we can
conveniently utilize the object ``m.p`` (itself an indexed ``Param`` object).

.. doctest::

  >>> # === Specify which parameters are uncertain ===
  >>> # We can pass IndexedParams this way to PyROS,
  >>> #   or as an expanded list per index
  >>> uncertain_parameters = [m.p]

.. note::
    Any ``Param`` object that is to be considered uncertain by PyROS
    must have the property ``mutable=True``.

PyROS will seek to identify solutions that remain feasible for any
realization of these parameters included in an uncertainty set.
To that end, we need to construct an
:class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`
object.
In our example, let us utilize the
:class:`~pyomo.contrib.pyros.uncertainty_sets.BoxSet`
constructor to specify
an uncertainty set of simple hyper-rectangular geometry.
For this, we will assume each parameter value is uncertain within a
percentage of its nominal value. Constructing this specific
:class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`
object can be done as follows:

.. doctest::

  >>> # === Define the pertinent data ===
  >>> relative_deviation = 0.15
  >>> bounds = [
  ...     (nominal_values[i] - relative_deviation*nominal_values[i],
  ...      nominal_values[i] + relative_deviation*nominal_values[i])
  ...     for i in range(4)
  ... ]

  >>> # === Construct the desirable uncertainty set ===
  >>> box_uncertainty_set = pyros.BoxSet(bounds=bounds)

Step 3: Solve with PyROS
^^^^^^^^^^^^^^^^^^^^^^^^^^

PyROS requires the user to supply one local and one global NLP solver to use
for solving sub-problems.
For convenience, we shall have PyROS invoke BARON as both the local and the
global NLP solver:

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> # === Designate local and global NLP solvers ===
  >>> local_solver = pyo.SolverFactory('baron')
  >>> global_solver = pyo.SolverFactory('baron')

.. note::
    Additional NLP optimizers can be automatically used in the event the primary
    subordinate local or global optimizer passed
    to the PyROS :meth:`~pyomo.contrib.pyros.PyROS.solve` method
    does not successfully solve a subproblem to an appropriate termination
    condition. These alternative solvers are provided through the optional
    keyword arguments ``backup_local_solvers`` and ``backup_global_solvers``.

The final step in solving a model with PyROS is to construct the
remaining required inputs, namely
``first_stage_variables`` and ``second_stage_variables``.
Below, we present two separate cases.

PyROS Termination Conditions
"""""""""""""""""""""""""""""

PyROS will return one of six termination conditions upon completion.
These termination conditions are defined through the
:class:`~pyomo.contrib.pyros.util.pyrosTerminationCondition` enumeration
and tabulated below.

.. table:: PyROS termination conditions.

    +----------------------------------------------------------------------------------+----------------------------------------------------------------+
    |  Termination Condition                                                           | Description                                                    |
    +==================================================================================+================================================================+
    |   :attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.robust_optimal`     |  The final solution is robust optimal                          |
    +----------------------------------------------------------------------------------+----------------------------------------------------------------+
    |   :attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.robust_feasible`    |  The final solution is robust feasible                         |
    +----------------------------------------------------------------------------------+----------------------------------------------------------------+
    |   :attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.robust_infeasible`  |  The posed problem is robust infeasible                        |
    +----------------------------------------------------------------------------------+----------------------------------------------------------------+
    |   :attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.max_iter`           |  Maximum number of GRCS iteration reached                      |
    +----------------------------------------------------------------------------------+----------------------------------------------------------------+
    |   :attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.time_out`           |  Maximum number of time reached                                |
    +----------------------------------------------------------------------------------+----------------------------------------------------------------+
    |   :attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.subsolver_error`    |  Unacceptable return status(es) from a user-supplied sub-solver|
    +----------------------------------------------------------------------------------+----------------------------------------------------------------+


A Single-Stage Problem
"""""""""""""""""""""""""
If we choose to designate all variables as either design or state variables,
without any control variables (i.e., all degrees of freedom are first-stage),
we can use PyROS to solve the single-stage problem as shown below.
In particular, let us instruct PyROS that variables
``m.x1`` through ``m.x6``, ``m.x19`` through ``m.x24``, and ``m.x31``
correspond to first-stage degrees of freedom.

.. _single-stage-problem:

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> # === Designate which variables correspond to first-stage
  >>> #     and second-stage degrees of freedom ===
  >>> first_stage_variables = [
  ...     m.x1, m.x2, m.x3, m.x4, m.x5, m.x6,
  ...     m.x19, m.x20, m.x21, m.x22, m.x23, m.x24, m.x31,
  ... ]
  >>> second_stage_variables = []
  >>> # The remaining variables are implicitly designated to be state variables

  >>> # === Call PyROS to solve the robust optimization problem ===
  >>> results_1 = pyros_solver.solve(
  ...     model=m,
  ...     first_stage_variables=first_stage_variables,
  ...     second_stage_variables=second_stage_variables,
  ...     uncertain_params=uncertain_parameters,
  ...     uncertainty_set=box_uncertainty_set,
  ...     local_solver=local_solver,
  ...     global_solver=global_solver,
  ...     objective_focus=pyros.ObjectiveType.worst_case,
  ...     solve_master_globally=True,
  ...     load_solution=False,
  ... )
  ==============================================================================
  PyROS: The Pyomo Robust Optimization Solver...
  ...
  ------------------------------------------------------------------------------
  Robust optimal solution identified.
  ------------------------------------------------------------------------------
  ...
  ------------------------------------------------------------------------------
  All done. Exiting PyROS.
  ==============================================================================
  >>> # === Query results ===
  >>> time = results_1.time
  >>> iterations = results_1.iterations
  >>> termination_condition = results_1.pyros_termination_condition
  >>> objective = results_1.final_objective_value
  >>> # === Print some results ===
  >>> single_stage_final_objective = round(objective,-1)
  >>> print(f"Final objective value: {single_stage_final_objective}")
  Final objective value: 48367380.0
  >>> print(f"PyROS termination condition: {termination_condition}")
  PyROS termination condition: pyrosTerminationCondition.robust_optimal

PyROS Results Object
"""""""""""""""""""""""""""
The results object returned by PyROS allows you to query the following information
from the solve call:

* ``iterations``: total iterations of the algorithm
* ``time``: total wallclock time (or elapsed time) in seconds
* ``pyros_termination_condition``: the GRCS algorithm termination condition
* ``final_objective_value``: the final objective function value.

The :ref:`preceding code snippet <single-stage-problem>`
demonstrates how to retrieve this information.

If we pass ``load_solution=True`` (the default setting)
to the :meth:`~pyomo.contrib.pyros.PyROS.solve` method,
then the solution at which PyROS terminates will be loaded to
the variables of the original deterministic model.
Note that in the :ref:`preceding code snippet <single-stage-problem>`,
we set ``load_solution=False`` to ensure the next set of runs shown here can
utilize the initial point loaded to the original deterministic model,
as the initial point may affect the performance of sub-solvers.

.. note::
    The reported ``final_objective_value`` and final model variable values
    depend on the selection of the option ``objective_focus``.
    The ``final_objective_value`` is the sum of first-stage
    and second-stage objective functions.
    If ``objective_focus = ObjectiveType.nominal``,
    second-stage objective and variables are evaluated at
    the nominal realization of the uncertain parameters, :math:`q^{\text{nom}}`.
    If ``objective_focus = ObjectiveType.worst_case``, second-stage objective
    and variables are evaluated at the worst-case realization
    of the uncertain parameters, :math:`q^{k^\ast}`
    where :math:`k^\ast = \mathrm{argmax}_{k \in \mathcal{K}}~f_2(x,z^k,y^k,q^k)`.


A Two-Stage Problem
""""""""""""""""""""""
For this next set of runs, we will
assume that some of the previously designated first-stage degrees of
freedom are in fact second-stage degrees of freedom.
PyROS handles second-stage degrees of freedom via the use of polynomial
decision rules, of which the degree is controlled through the
optional keyword argument ``decision_rule_order`` to the PyROS
:meth:`~pyomo.contrib.pyros.PyROS.solve` method.
In this example, we select affine decision rules by setting
``decision_rule_order=1``:

.. _example-two-stg:

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> # === Define the variable partitioning
  >>> first_stage_variables =[m.x5, m.x6, m.x19, m.x22, m.x23, m.x24, m.x31]
  >>> second_stage_variables = [m.x1, m.x2, m.x3, m.x4, m.x20, m.x21]
  >>> # The remaining variables are implicitly designated to be state variables

  >>> # === Call PyROS to solve the robust optimization problem ===
  >>> results_2 = pyros_solver.solve(
  ...     model=m,
  ...     first_stage_variables=first_stage_variables,
  ...     second_stage_variables=second_stage_variables,
  ...     uncertain_params=uncertain_parameters,
  ...     uncertainty_set=box_uncertainty_set,
  ...     local_solver=local_solver,
  ...     global_solver=global_solver,
  ...     objective_focus=pyros.ObjectiveType.worst_case,
  ...     solve_master_globally=True,
  ...     decision_rule_order=1,
  ... )
  ==============================================================================
  PyROS: The Pyomo Robust Optimization Solver...
  ...
  ------------------------------------------------------------------------------
  Robust optimal solution identified.
  ------------------------------------------------------------------------------
  ...
  ------------------------------------------------------------------------------
  All done. Exiting PyROS.
  ==============================================================================
  >>> # === Compare final objective to the single-stage solution
  >>> two_stage_final_objective = round(
  ...     pyo.value(results_2.final_objective_value),
  ...     -1,
  ... )
  >>> percent_difference = 100 * (
  ...     two_stage_final_objective - single_stage_final_objective
  ... ) / (single_stage_final_objective)
  >>> print("Percent objective change relative to constant decision rules "
  ...       f"objective: {percent_difference:.2f}")
  Percent objective change relative to constant decision rules objective: -24...

For this example, we notice a ~25% decrease in the final objective
value when switching from a static decision rule (no second-stage recourse)
to an affine decision rule.


Specifying Arguments Indirectly Through ``options``
"""""""""""""""""""""""""""""""""""""""""""""""""""
Like other Pyomo solver interface methods,
:meth:`~pyomo.contrib.pyros.PyROS.solve`
provides support for specifying options indirectly by passing
a keyword argument ``options``, whose value must be a :class:`dict`
mapping names of arguments to :meth:`~pyomo.contrib.pyros.PyROS.solve`
to their desired values.
For example, the ``solve()`` statement in the
:ref:`two-stage problem snippet <example-two-stg>`
could have been equivalently written as:

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> results_2 = pyros_solver.solve(
  ...     model=m,
  ...     first_stage_variables=first_stage_variables,
  ...     second_stage_variables=second_stage_variables,
  ...     uncertain_params=uncertain_parameters,
  ...     uncertainty_set=box_uncertainty_set,
  ...     local_solver=local_solver,
  ...     global_solver=global_solver,
  ...     options={
  ...         "objective_focus": pyros.ObjectiveType.worst_case,
  ...         "solve_master_globally": True,
  ...         "decision_rule_order": 1,
  ...     },
  ... )
  ==============================================================================
  PyROS: The Pyomo Robust Optimization Solver...
  ...
  ------------------------------------------------------------------------------
  Robust optimal solution identified.
  ------------------------------------------------------------------------------
  ...
  ------------------------------------------------------------------------------
  All done. Exiting PyROS.
  ==============================================================================

In the event an argument is passed directly
by position or keyword, *and* indirectly through ``options``,
an appropriate warning is issued,
and the value passed directly takes precedence over the value
passed through ``options``.


The Price of Robustness
""""""""""""""""""""""""
In conjunction with standard Python control flow tools,
PyROS facilitates a "price of robustness" analysis for a model of interest
through the evaluation and comparison of the robust optimal
objective function value across any appropriately constructed hierarchy
of uncertainty sets.
In this example, we consider a sequence of
box uncertainty sets centered on the nominal uncertain
parameter realization, such that each box is parameterized
by a real value specifying a relative box size.
To this end, we construct an iterable called ``relative_deviation_list``
whose entries are ``float`` values representing the relative sizes.
We then loop through ``relative_deviation_list`` so that for each relative
size, the corresponding robust optimal objective value
can be evaluated by creating an appropriate
:class:`~pyomo.contrib.pyros.uncertainty_sets.BoxSet`
instance and invoking the PyROS solver:

.. code::

  >>> # This takes a long time to run and therefore is not a doctest
  >>> # === An array of maximum relative deviations from the nominal uncertain
  >>> #     parameter values to utilize in constructing box sets
  >>> relative_deviation_list = [0.00, 0.10, 0.20, 0.30, 0.40]
  >>> # === Final robust optimal objectives
  >>> robust_optimal_objectives = []
  >>> for relative_deviation in relative_deviation_list: # doctest: +SKIP
  ...     bounds = [
  ...         (nominal_values[i] - relative_deviation*nominal_values[i],
  ...          nominal_values[i] + relative_deviation*nominal_values[i])
  ...         for i in range(4)
  ...     ]
  ...     box_uncertainty_set = pyros.BoxSet(bounds = bounds)
  ...     results = pyros_solver.solve(
  ...         model=m,
  ...         first_stage_variables=first_stage_variables,
  ...         second_stage_variables=second_stage_variables,
  ...         uncertain_params=uncertain_parameters,
  ...         uncertainty_set= box_uncertainty_set,
  ...         local_solver=local_solver,
  ...         global_solver=global_solver,
  ...         objective_focus=pyros.ObjectiveType.worst_case,
  ...         solve_master_globally=True,
  ...         decision_rule_order=1,
  ...     )
  ...     is_robust_optimal = (
  ...         results.pyros_termination_condition
  ...         == pyros.pyrosTerminationCondition.robust_optimal
  ...     )
  ...     if not is_robust_optimal:
  ...         print(f"Instance for relative deviation: {relative_deviation} "
  ...               "not solved to robust optimality.")
  ...         robust_optimal_objectives.append("-----")
  ...     else:
  ...         robust_optimal_objectives.append(str(results.final_objective_value))

For this example, we obtain the following price of robustness results:

.. table:: Price of robustness results.

    +------------------------------------------+------------------------------+-----------------------------+
    |   Uncertainty Set Size (+/-) :sup:`o`    | Robust Optimal Objective     | % Increase :sup:`x`         |
    +==========================================+==============================+=============================+
    |   0.00                                   | 35,837,659.18                | 0.00 %                      |
    +------------------------------------------+------------------------------+-----------------------------+
    |   0.10                                   | 36,135,182.66                | 0.83 %                      |
    +------------------------------------------+------------------------------+-----------------------------+
    |   0.20                                   | 36,437,979.81                | 1.68 %                      |
    +------------------------------------------+------------------------------+-----------------------------+
    |   0.30                                   | 43,478,190.91                | 21.32 %                     |
    +------------------------------------------+------------------------------+-----------------------------+
    |   0.40                                   | ``robust_infeasible``        | :math:`\text{-----}`        |
    +------------------------------------------+------------------------------+-----------------------------+

Notice that PyROS was successfully able to determine the robust
infeasibility of the problem under the largest uncertainty set.

:sup:`o` **Relative Deviation from Nominal Realization**

:sup:`x` **Relative to Deterministic Optimal Objective**

This example clearly illustrates the potential impact of the uncertainty
set size on the robust optimal objective function value
and demonstrates the ease of implementing a price of robustness study
for a given optimization problem under uncertainty.

PyROS Solver Log Output
-------------------------------

The PyROS solver log output is controlled through the optional
``progress_logger`` argument, itself cast to
a standard Python logger (:py:class:`logging.Logger`) object
at the outset of a :meth:`~pyomo.contrib.pyros.PyROS.solve` call.
The level of detail of the solver log output
can be adjusted by adjusting the level of the
logger object; see :ref:`the following table <table-logging-levels>`.
Note that by default, ``progress_logger`` is cast to a logger of level
:py:obj:`logging.INFO`.

We refer the reader to the
:doc:`official Python logging library documentation <python:library/logging>`
for customization of Python logger objects;
for a basic tutorial, see the :doc:`logging HOWTO <python:howto/logging>`.

.. _table-logging-levels:

.. list-table:: PyROS solver log output at the various standard Python :py:mod:`logging` levels.
   :widths: 10 50
   :header-rows: 1

   * - Logging Level
     - Output Messages
   * - :py:obj:`logging.ERROR`
     - * Information on the subproblem for which an exception was raised
         by a subordinate solver
       * Details about failure of the PyROS coefficient matching routine
   * - :py:obj:`logging.WARNING`
     - * Information about a subproblem not solved to an acceptable status
         by the user-provided subordinate optimizers
       * Invocation of a backup solver for a particular subproblem
       * Caution about solution robustness guarantees in event that
         user passes ``bypass_global_separation=True``
   * - :py:obj:`logging.INFO`
     - * PyROS version, author, and disclaimer information
       * Summary of user options
       * Breakdown of model component statistics
       * Iteration log table
       * Termination details: message, timing breakdown, summary of statistics
   * - :py:obj:`logging.DEBUG`
     - * Termination outcomes and summary of statistics for
         every master feasility, master, and DR polishing problem
       * Progress updates for the separation procedure
       * Separation subproblem initial point infeasibilities
       * Summary of separation loop outcomes: performance constraints
         violated, uncertain parameter scenario added to the
         master problem
       * Uncertain parameter scenarios added to the master problem
         thus far

An example of an output log produced through the default PyROS
progress logger is shown in
:ref:`the snippet that follows <solver-log-snippet>`.
Observe that the log contains the following information:


* **Introductory information** (lines 1--18).
  Includes the version number, author
  information, (UTC) time at which the solver was invoked,
  and, if available, information on the local Git branch and
  commit hash.
* **Summary of solver options** (lines 19--38).
* **Preprocessing information** (lines 39--41).
  Wall time required for preprocessing
  the deterministic model and associated components,
  i.e. standardizing model components and adding the decision rule
  variables and equations.
* **Model component statistics** (lines 42--58).
  Breakdown of model component statistics.
  Includes components added by PyROS, such as the decision rule variables
  and equations.
* **Iteration log table** (lines 59--69).
  Summary information on the problem iterates and subproblem outcomes.
  The constituent columns are defined in detail in
  :ref:`the table following the snippet <table-iteration-log-columns>`.
* **Termination message** (lines 70--71). Very brief summary of the termination outcome.
* **Timing statistics** (lines 72--88).
  Tabulated breakdown of the solver timing statistics, based on a
  :class:`pyomo.common.timing.HierarchicalTimer` printout.
  The identifiers are as follows:

  * ``main``: Total time elapsed by the solver.
  * ``main.dr_polishing``: Total time elapsed by the subordinate solvers
    on polishing of the decision rules.
  * ``main.global_separation``: Total time elapsed by the subordinate solvers
    on global separation subproblems.
  * ``main.local_separation``: Total time elapsed by the subordinate solvers
    on local separation subproblems.
  * ``main.master``: Total time elapsed by the subordinate solvers on
    the master problems.
  * ``main.master_feasibility``: Total time elapsed by the subordinate solvers
    on the master feasibility problems.
  * ``main.preprocessing``: Total preprocessing time.
  * ``main.other``: Total overhead time.

* **Termination statistics** (lines 89--94). Summary of statistics related to the
  iterate at which PyROS terminates.
* **Exit message** (lines 95--96).


.. _solver-log-snippet:

.. code-block:: text
   :caption: PyROS solver output log for the :ref:`two-stage problem example <example-two-stg>`.
   :linenos:

   ==============================================================================
   PyROS: The Pyomo Robust Optimization Solver, v1.2.11.
          Pyomo version: 6.7.2
          Commit hash: unknown
          Invoked at UTC 2024-03-28T00:00:00.000000

   Developed by: Natalie M. Isenberg (1), Jason A. F. Sherman (1),
                 John D. Siirola (2), Chrysanthos E. Gounaris (1)
   (1) Carnegie Mellon University, Department of Chemical Engineering
   (2) Sandia National Laboratories, Center for Computing Research

   The developers gratefully acknowledge support from the U.S. Department
   of Energy's Institute for the Design of Advanced Energy Systems (IDAES).
   ==============================================================================
   ================================= DISCLAIMER =================================
   PyROS is still under development. 
   Please provide feedback and/or report any issues by creating a ticket at
   https://github.com/Pyomo/pyomo/issues/new/choose
   ==============================================================================
   Solver options:
    time_limit=None
    keepfiles=False
    tee=False
    load_solution=True
    symbolic_solver_labels=False
    objective_focus=<ObjectiveType.worst_case: 1>
    nominal_uncertain_param_vals=[0.13248000000000001, 4.97, 4.97, 1800]
    decision_rule_order=1
    solve_master_globally=True
    max_iter=-1
    robust_feasibility_tolerance=0.0001
    separation_priority_order={}
    progress_logger=<PreformattedLogger pyomo.contrib.pyros (INFO)>
    backup_local_solvers=[]
    backup_global_solvers=[]
    subproblem_file_directory=None
    bypass_local_separation=False
    bypass_global_separation=False
    p_robustness={}
   ------------------------------------------------------------------------------
   Preprocessing...
   Done preprocessing; required wall time of 0.175s.
   ------------------------------------------------------------------------------
   Model statistics:
     Number of variables : 62
       Epigraph variable : 1
       First-stage variables : 7
       Second-stage variables : 6
       State variables : 18
       Decision rule variables : 30
     Number of uncertain parameters : 4
     Number of constraints : 81
       Equality constraints : 24
         Coefficient matching constraints : 0
         Decision rule equations : 6
         All other equality constraints : 18
       Inequality constraints : 57
         First-stage inequalities (incl. certain var bounds) : 10
         Performance constraints (incl. var bounds) : 47
   ------------------------------------------------------------------------------
   Itn  Objective    1-Stg Shift  2-Stg Shift  #CViol  Max Viol     Wall Time (s)
   ------------------------------------------------------------------------------
   0     3.5838e+07  -            -            5       1.8832e+04   1.741        
   1     3.5838e+07  3.5184e-15   3.9404e-15   10      4.2516e+06   3.766        
   2     3.5993e+07  1.8105e-01   7.1406e-01   13      5.2004e+06   6.288
   3     3.6285e+07  5.1968e-01   7.7753e-01   4       1.7892e+04   8.247
   4     3.6285e+07  9.1166e-13   1.9702e-15   0       7.1157e-10g  11.456
   ------------------------------------------------------------------------------
   Robust optimal solution identified.
   ------------------------------------------------------------------------------
   Timing breakdown:

   Identifier                ncalls   cumtime   percall      %
   -----------------------------------------------------------
   main                           1    11.457    11.457  100.0
        ------------------------------------------------------
        dr_polishing              4     0.682     0.171    6.0
        global_separation        47     1.109     0.024    9.7
        local_separation        235     5.810     0.025   50.7
        master                    5     1.353     0.271   11.8
        master_feasibility        4     0.247     0.062    2.2
        preprocessing             1     0.429     0.429    3.7
        other                   n/a     1.828       n/a   16.0
        ======================================================
   ===========================================================

   ------------------------------------------------------------------------------
   Termination stats:
    Iterations            : 5
    Solve time (wall s)   : 11.457
    Final objective value : 3.6285e+07
    Termination condition : pyrosTerminationCondition.robust_optimal
   ------------------------------------------------------------------------------
   All done. Exiting PyROS.
   ==============================================================================


The iteration log table is designed to provide, in a concise manner,
important information about the progress of the iterative algorithm for
the problem of interest.
The constituent columns are defined in the
:ref:`table that follows <table-iteration-log-columns>`.

.. _table-iteration-log-columns:

.. list-table:: PyROS iteration log table columns.
   :widths: 10 50
   :header-rows: 1

   * - Column Name
     - Definition
   * - Itn
     - Iteration number.
   * - Objective
     - Master solution objective function value.
       If the objective of the deterministic model provided
       has a maximization sense,
       then the negative of the objective function value is displayed.
       Expect this value to trend upward as the iteration number
       increases.
       If the master problems are solved globally
       (by passing ``solve_master_globally=True``),
       then after the iteration number exceeds the number of uncertain parameters,
       this value should be monotonically nondecreasing
       as the iteration number is increased.
       A dash ("-") is produced in lieu of a value if the master
       problem of the current iteration is not solved successfully.
   * - 1-Stg Shift
     - Infinity norm of the relative difference between the first-stage
       variable vectors of the master solutions of the current
       and previous iterations. Expect this value to trend
       downward as the iteration number increases.
       A dash ("-") is produced in lieu of a value
       if the current iteration number is 0,
       there are no first-stage variables,
       or the master problem of the current iteration is not solved successfully.
   * - 2-Stg Shift
     - Infinity norm of the relative difference between the second-stage
       variable vectors (evaluated subject to the nominal uncertain
       parameter realization) of the master solutions of the current
       and previous iterations. Expect this value to trend
       downward as the iteration number increases.
       A dash ("-") is produced in lieu of a value
       if the current iteration number is 0,
       there are no second-stage variables,
       or the master problem of the current iteration is not solved successfully.
   * - #CViol
     - Number of performance constraints found to be violated during
       the separation step of the current iteration.
       Unless a custom prioritization of the model's performance constraints
       is specified (through the ``separation_priority_order`` argument),
       expect this number to trend downward as the iteration number increases.
       A "+" is appended if not all of the separation problems
       were solved successfully, either due to custom prioritization, a time out,
       or an issue encountered by the subordinate optimizers.
       A dash ("-") is produced in lieu of a value if the separation
       routine is not invoked during the current iteration.
   * - Max Viol
     - Maximum scaled performance constraint violation.
       Expect this value to trend downward as the iteration number increases.
       A 'g' is appended to the value if the separation problems were solved
       globally during the current iteration.
       A dash ("-") is produced in lieu of a value if the separation
       routine is not invoked during the current iteration, or if there are
       no performance constraints.
   * - Wall time (s)
     - Total time elapsed by the solver, in seconds, up to the end of the
       current iteration.


Feedback and Reporting Issues
-------------------------------
Please provide feedback and/or report any problems by opening an issue on
the `Pyomo GitHub page <https://github.com/Pyomo/pyomo/issues/new/choose>`_.
