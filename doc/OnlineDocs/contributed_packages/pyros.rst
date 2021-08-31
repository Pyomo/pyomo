############
PyROS Solver
############

PyROS (Pyomo Robust Optimization Solver) is a metasolver capability within Pyomo for solving non-convex,
two-stage optimization models using adjustable robust optimization.

It was developed by **Natalie M. Isenberg** and **Chrysanthos E. Gounaris** of Carnegie Mellon University,
in collaboration with **John D. Siirola** of Sandia National Labs.
The developers gratefully acknowledge support from the U.S. Department of Energy's
`Institute for the Design of Advanced Energy Systems (IDAES) <https://idaes.org>`_.

Methodology Overview
-----------------------------

Below is an overview of the type of optimization models PyROS can accomodate.


* PyROS is suitable for optimization models of **continuous variables** that may feature non-linearities (including **non-convexities**) in both the variables and uncertain parameters.
* PyROS can handle **equality constraints** defining state variables, including implicit state variables that cannot be eliminated via reformulation.
* PyROS allows for **two-stage** optimization problems that may feature both first-stage and second-stage degrees of freedom.

The general form of a deterministic optimization problem that can be passed into PyROS is shown below:

.. math::
    \begin{align*}
    \displaystyle \min_{\substack{x \in \mathcal{X}, \\ z \in \mathbb{R}^n, y\in\mathbb{R}^a}} & ~~ f_1\left(x\right) + f_2\left(x,z,y; q^0\right) & \\
    \displaystyle \text{s.t.} \quad \: & ~~ g_i\left(x, z, y; q^0\right) \leq 0 & \forall i \in \mathcal{I} \\
    & ~~ h_j\left(x,z,y; q^0\right) = 0 & \forall j \in \mathcal{J} \\
    \end{align*}

where:

* :math:`x \in \mathcal{X}` are the "design" variables (i.e., first-stage degrees of freedom), where :math:`\mathcal{X} \subseteq \mathbb{R}^m` is the feasible space defined by the model constraints that only reference these variables
* :math:`z \in \mathbb{R}^n` are the "control" variables (i.e., second-stage degrees of freedom)
* :math:`y \in \mathbb{R}^a` are the "state" variables
* :math:`q \in \mathbb{R}^w` is the vector of parameters that we shall later consider to be uncertain, and :math:`q^0` is the vector of nominal values associated with those.
* :math:`f_1\left(x\right)` are the terms of the objective function that depend only on design variables
* :math:`f_2\left(x, z, y; q\right)` are the terms of the objective function that depend on control and/or state variables
* :math:`g_i\left(x, z, y; q\right)` is the :math:`i^\text{th}` inequality constraint in set :math:`\mathcal{I}` (see Note)
* :math:`h_j\left(x, z, y; q\right)` is the :math:`j^\text{th}` equality constraint in set :math:`\mathcal{J}` (see Note)

.. note::
    * Applicable bounds on variables :math:`z` and/or :math:`y` are assumed to have been incorporated in the set of inequality constraints :math:`\mathcal{I}`.
    * A key requirement of PyROS is that each value of :math:`\left(x, z, q \right)` maps to a unique value of :math:`y`, a property that is assumed to be properly enforced by the system of equality constraints :math:`\mathcal{J}`. If such unique mapping does not hold, then the selection of 'state' (i.e., not degree of freedom) variables :math:`y` is incorrect, and one or more of the :math:`y` variables should be appropriately redesignated to be part of either :math:`x` or :math:`z`.

In order to cast the robust optimization counterpart formulation of the above model, we shall now assume that the uncertain parameters may attain
any realization from within an uncertainty set :math:`\mathcal{Q} \subseteq \mathbb{R}^w`, such that :math:`q^0 \in \mathcal{Q}`.
The set :math:`\mathcal{Q}` is assumed to be closed and bounded, while it can be **either continuous or discrete**.

Based on the above notation, the form of the robust counterpart addressed in PyROS is shown below:

.. math::
    \begin{align*}
    \displaystyle \min_{x \in \mathcal{X}}
    & \displaystyle \max_{q \in \mathcal{Q}}
    & \displaystyle \min_{z \in \mathbb{R}^n, y \in \mathbb{R}^a} \ \ & \displaystyle ~~ f_1\left(x\right) + f_2\left(x, z, y, q\right) & & \\
    & & \text{s.t.} \quad \:& \displaystyle ~~ g_i\left(x, z, y, q\right) \leq 0 &  & \forall i \in \mathcal{I}\\
    & & & \displaystyle ~~ h_j\left(x, z, y, q\right) = 0 &  & \forall j \in \mathcal{J}
    \end{align*}

In order to solve problems of the above type, PyROS implements the
Generalized Robust Cutting-Set algorithm developed in [GRCSPaper]_.

When using PyROS, please consider citing the above paper.

PyROS Required Inputs
-----------------------------
The required inputs to the PyROS solver are the following:

* The determinisitic optimization model
* List of first-stage ("design") variables
* List of second-stage ("control") variables
* List of parameters to be considered uncertain
* The uncertainty set
* Subordinate local and global NLP optimization solvers

Below is a list of arguments that PyROS expects the user to provide when calling the ``solve`` command.
Note how all but the ``model`` argument **must** be specified as ``kwargs``.

model : ``ConcreteModel``
    A ``ConcreteModel`` object representing the deterministic model.
first_stage_variables : ``list(Var)``
    A list of Pyomo ``Var`` objects representing the first-stage degrees of freedom (design variables) in ``model``.
second_stage_variables : ``list(Var)``
    A list of Pyomo ``Var`` objects representing second-stage degrees of freedom (control variables) in ``model``.
uncertain_params : ``list(Param)``
    A list of Pyomo ``Param`` objects in ``deterministic_model`` to be considered uncertain. These specified ``Param`` objects must have the property ``mutable=True``.
uncertainty_set : ``UncertaintySet``
    A PyROS ``UncertaintySet`` object representing uncertainty in the space of those parameters listed in the ``uncertain_params`` object.
local_solver : ``Solver``
    A Pyomo ``Solver`` instance for a local NLP optimization solver.
global_solver : ``Solver``
    A Pyomo ``Solver`` instance for a global NLP optimization solver.

.. note::
    Any variables in the model not specified to be first- or second-stage variables are automatically considered to be state variables.

PyROS Solver Interface
-----------------------------

.. autoclass:: pyomo.contrib.pyros.PyROS
    :members: solve

.. note::
    Solving the master problems globally (via option ``solve_masters_globally=True``) is one of the requirements to guarantee robust optimality;
    solving the master problems locally can only lead to a robust feasible solution.

.. note::
    Selecting worst-case objective (via option ``objective_focus=ObjectiveType.worst_case``) is one of the requirements to guarantee robust optimality;
    selecting nominal objective can only lead to a robust feasible solution,
    albeit one that has optimized the sum of first- and (nominal) second-stage objectives.

.. note::
    To utilize option ``p_robustness``, a dictionary of the following form must be supplied via the ``kwarg``:
    There must be a key (``str``) called 'rho', which maps to a non-negative value, where '1+rho' defines a bound
    for the ratio of the objective that any scenario may exhibit compared to the nominal objective.

PyROS Uncertainty Sets
-----------------------------
PyROS contains pre-implemented ``UncertaintySet`` specializations for many types of commonly used uncertainty sets.
Additional capabilities for intersecting multiple PyROS ``UncertaintySet`` objects so as to create custom sets are also provided
via the ``IntersectionSet`` class.  Custom user-specified sets can also be defined via the base ``UncertaintySet`` class.

Mathematical representations of the sets are shown below, followed by the class descriptions.

.. list-table:: PyROS Uncertainty Sets
   :header-rows: 1
   :class: tight-table

   * - Uncertainty Set Type
     - Set Representation
   * - ``BoxSet``
     - :math:`Q_X = \left\{q \in \mathbb{R}^n : q^\ell \leq q \leq q^u\right\} \\ q^\ell \in \mathbb{R}^n \\ q^u \in \mathbb{R}^n : \left\{q^\ell \leq q^u\right\}`
   * - ``CardinalitySet``
     - :math:`Q_C = \left\{q \in \mathbb{R}^n : q = q^0 + (\hat{q} \circ \xi) \text{ for some } \xi \in \Xi_C\right\}\\ \Xi_C = \left\{\xi \in [0, 1]^n : \displaystyle\sum_{i=1}^{n} \xi_i \leq \Gamma\right\} \\ \Gamma \in [0, n] \\ \hat{q} \in \mathbb{R}^{n}_{+} \\ q^0 \in \mathbb{R}^n`
   * - ``BudgetSet``
     - :math:`Q_B = \left\{q \in \mathbb{R}^n_+: \displaystyle\sum_{i \in B_\ell} q_i \leq b_\ell \ \forall \ell \in \left\{1,\ldots,L\right\} \right\} \\ b_\ell \in \mathbb{R}^{L}_+`
   * - ``FactorModelSet``
     - :math:`Q_F = \left\{q \in \mathbb{R}^n: \displaystyle q = q^0 + \Psi \xi \text{ for some }\xi \in \Xi_F\right\} \\ \Xi_F = \left\{ \xi \in \left[-1, 1\right]^F, \left\lvert \displaystyle \sum_{f=1}^{F} \xi_f\right\rvert \leq \beta F \right\} \\ \beta \in [0,1] \\ \Psi \in \mathbb{R}^{n \times F}_+ \\ q^0 \in \mathbb{R}^n`
   * - ``PolyhedralSet``
     - :math:`Q_P = \left\{q \in \mathbb{R}^n: \displaystyle A q \leq b \right\} \\ A \in \mathbb{R}^{m \times n} \\ b \in \mathbb{R}^{m} \\ q^0 \in \mathbb{R}^n: {Aq^0 \leq b}`
   * - ``AxisAlignedEllipsoidalSet``
     - :math:`Q_A = \left\{q \in \mathbb{R}^n: \displaystyle \sum\limits_{i=1 : \atop \left\{ \alpha_i > 0 \right\} } \left(\frac{q_i - q_i^0}{\alpha_i} \right)^2 \leq 1 , \quad q_i = q^0_i \quad \forall i : \left\{\alpha_i=0\right\}\right\} \\ \alpha \in \mathbb{R}^n_+, \\ q^0 \in \mathbb{R}^n`
   * - ``EllipsoidalSet``
     - :math:`Q_E = \left\{q \in \mathbb{R}^n: \displaystyle q = q^0 + P^{1/2} \xi \text{ for some } \xi \in \Xi_E \right\} \\ \Xi_E = \left\{\xi \in \mathbb{R} : \xi^T\xi \leq s \right\} \\ P \in \mathbb{S}^{n\times n}_+ \\ s \in \mathbb{R}_+ \\ q^0 \in \mathbb{R}^n`
   * - ``UncertaintySet``
     - :math:`Q_U = \left\{q \in \mathbb{R}^n: \displaystyle g_i(q) \leq 0 \quad \forall i \in \left\{1,\ldots,m \right\}\right\} \\ m \in \mathbb{N}_+ \\ g_i : \mathbb{R}^n \mapsto \mathbb{R} \, \forall i \in \left\{1,\ldots,m\right\}, \\ q^0 \in \mathbb{R}^n : \left\{g_i(q^0) \leq 0  \ \forall i \in \left\{1,\ldots,m\right\}\right\}`
   * - ``DiscreteScenariosSet``
     - :math:`Q_D = \left\{q^s : s = 0,\ldots,D \right\} \\ D \in \mathbb{N} \\ q^s \in \mathbb{R}^n \forall s \in \left\{ 0,\ldots,D\right\}`
   * - ``IntersectionSet``
     - :math:`Q_I = \left\{q \in \mathbb{R}^n: \displaystyle q \in \bigcap_{i \in \left\{1,\ldots,m\right\}} Q_i\right\} \\ Q_i \subset \mathbb{R}^n \quad \forall i \in \left\{1,\ldots,m\right\}`

.. note::
    Each of the PyROS uncertainty set classes inherits from the ``UncertaintySet`` base class.

PyROS Uncertainty Set Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.BoxSet
    :special-members: __init__, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.CardinalitySet
    :special-members: __init__, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.BudgetSet
    :special-members: __init__, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.FactorModelSet
    :special-members: __init__, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.PolyhedralSet
    :special-members: __init__, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.AxisAlignedEllipsoidalSet
    :special-members: __init__, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.EllipsoidalSet
    :special-members: __init__, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.UncertaintySet
    :special-members: __init__, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.DiscreteScenarioSet
    :special-members: __init__, parameter_bounds, dim, point_in_set

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.IntersectionSet
    :special-members: __init__, parameter_bounds, dim, point_in_set


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
