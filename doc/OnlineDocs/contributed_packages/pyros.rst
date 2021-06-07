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
* PyROS allows for **two-stage** optimization problems that may feauture both first-stage and second-stage degrees of freedom.

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

In order to solve problems of the above type, PyROS implements the Generalized Robust Cutting-Set algorithm developed in:

`Isenberg, NM, Akula, P, Eslick, JC, Bhattacharyya, D, Miller, DC, Gounaris, CE. A generalized cutting‐set approach for nonlinear robust optimization in process systems engineering. AIChE J. 2021; 67:e17175. DOI 10.1002/aic.17175 <https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.17175>`_

When using PyROS, please consider citing the above paper.

PyROS Required Inputs
-----------------------------
The required inputs to the PyROS solver are the following:

* The determinisitic optimization model
* List of first-stage "design" variables
* List of second-stage "control" variables
* List of parameters to be considered uncertain
* The uncertainty set
* Subordinate local and global NLP optimization solvers

.. note::
    Any variables in the model not specified to be first- or second-stage variables are automatically considered to be state variables.

Below is a list of arguments that PyROS expects the user to provide when calling the ``solve`` command.
Note how all but the ``model`` argument **must** be specified as ``kwargs``.

model : ``ConcreteModel``
    A ``ConcreteModel`` object representing the deterministic model
first_stage_variables : ``list(Var)``
    A list of Pyomo ``Var`` objects representing the first-stage degrees of freedom (design variables) in ``model``
second_stage_variables : ``list(Var)``
    A list of Pyomo ``Var`` objects representing second-stage degrees of freedom (control variables) in ``model``
uncertain_params : ``list(Param)``
    A list of Pyomo ``Param`` objects in ``deterministic_model`` to be considered uncertain. These specified ``Param`` objects must have the property ``mutable=True``
uncertainty_set : ``UncertaintySet``
    A PyROS ``UncertaintySet`` object that represents the uncertainty in the space of the specified ``uncertain_param`` objects
local_solver : ``Solver``
    A Pyomo ``Solver`` instance for a local NLP optimization solver
global_solver : ``Solver``
    A Pyomo ``Solver`` instance for a global NLP optimization solver

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
     - :math:`Q_U = \left\{q \in \mathbb{R}^n: \displaystyle g_i(q) \leq 0 \quad \forall i \in \left\{1,\ldots,m \right\}\right\} \\ m \in \mathbb{N}_+ \\ g_i : \mathbb{R}^n \mapsto \mathbb{R} \\ \forall i \in \left\{1,\ldots,m\right\}, \\ q^0 \in \mathbb{R}^n : \left\{g_i(q^0) \leq 0  \ \forall i \in \left\{1,\ldots,m\right\}\right\}`
   * - ``DiscreteSet``
     - :math:`Q_D = \left\{q^s : s = 0,\ldots,D \right\} \\ D \in \mathbb{N} \\ q^s \in \mathbb{R}^n \forall s \in \left\{ 0,\ldots,D\right\}`
   * - ``IntersectionSet``
     - :math:`Q_I = \left\{q \in \mathbb{R}^n: \displaystyle q \in \bigcap_{i \in \left\{1,\ldots,m\right\}} Q_i\right\} \\ Q_i \subset \mathbb{R}^n \quad \forall i \in \left\{1,\ldots,m\right\}`

.. note::
    Each of the PyROS uncertainty set classes inherits from the ``UncertaintySet`` base class.

PyROS Uncertainty Set Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.BoxSet
    :special-members: __init__

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.CardinalitySet
    :special-members: __init__

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.BudgetSet
    :special-members: __init__

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.FactorModelSet
    :special-members: __init__

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.PolyhedralSet
    :special-members: __init__

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.AxisAlignedEllipsoidalSet
    :special-members: __init__

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.EllipsoidalSet
    :special-members: __init__

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.UncertaintySet
    :special-members: __init__

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.DiscreteSet
    :special-members: __init__

.. autoclass:: pyomo.contrib.pyros.uncertainty_sets.IntersectionSet
    :special-members: __init__


PyROS Usage Example
-----------------------------

We will use an example to illustrate the usage of PyROS. The problem we will use is called *hydro* and comes from
the GAMS example problem database in `The GAMS Model Library <https://www.gams.com/latest/gamslib_ml/libhtml/>`_. The model was converted to Pyomo format via the `GAMS Convert tool <https://www.gams.com/latest/docs/S_CONVERT.html>`_.

Per our analysis, this model is a QCQP with 13 variables representing degrees of freedom, as well as an additional 18 state variables.
This model features 6 linear inequality constraints, 6 linear equality constraints, 6 non-linear (quadratic) equalities, and a quadratic objective.
We have augmented this model by converting one objective coefficient, two constraint coefficients, and one constraint right-hand side into Param objects so that they can be considered uncertain later on.

.. note::
    We can easily verify that this model satisfies the requirement that each value of :math:`\left(x, z, q \right)` maps to a unique value of :math:`y`, which indicates a proper partition of variables between (first- or second-stage) degrees of freedom and state variables.

Step 1: Define the Deterministic Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The deterministic Pyomo model for *hydro* is shown below. For purposes of our robust optimization example,
we assume that some of the data are potentially uncertain and utilize the ``m.p`` object (itself an indexed ``Param`` object)
to instruct PyROS to consider as uncertain in the model.

.. note::
    Primitive data (python literals) that have been hard-coded within a deterministic model cannot be considered uncertain,
    unless they are first converted to ``Param`` objects within the ``model`` object.

.. note::
    Any ``Param`` object that is to be considered uncertain (by being specified as ``uncertain_params``) must have the property ``mutable=True``.

.. code::

  >>> # === Required imports ===
  >>> import pyomo.environ as pyo
  >>> import pyomo.pyros as pyros

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

  >>> # === Uncertain Params ===
  >>> m.set_of_params = pyo.Set(initialize=[0, 1, 2, 3])
  >>> nominal_values = {0:82.8*0.0016, 1:4.97, 2:4.97, 3:1800}
  >>> m.p = pyo.Param(m.set_of_params, initialize=nominal_values, mutable=True)

  >>> # === Specify the objective function ===
  >>> m.obj = pyo.Objective(expr=m.p[0]*m.x1**2 + 82.8*8*m.x1 + 82.8*0.0016*m.x2**2 +
                                 82.8*82.8*8*m.x2 + 82.8*0.0016*m.x3**2 + 82.8*8*m.x3 +
                                 82.8*0.0016*m.x4**2 + 82.8*8*m.x4 + 82.8*0.0016*m.x5**2 +
                                 82.8*8*m.x5 + 82.8*0.0016*m.x6**2 + 82.8*8*m.x6 + 248400,
                                 sense=pyo.minimize)

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Given uncertainty in the model parameters ``(m.p[0], m.p[1], m.p[2], m.p[3])``, we can use PyROS to
identify solutions that remain feasible for any realization of these parameters included in an uncertainty set that we shall postulate.
Furthermore, we can utilize PyROS to show the robust infeasibility of the deterministic solution in light of our proposed
uncertainty set.

We will define a ``UncertaintySet`` object to construct a representative uncertainty set in the space of our parameters.
For this example, let us utilize the ``BoxSet`` constructor to specify an uncertainty set of simple hyper-rectangular geometry.
For this, we will assume each parameter value is uncertain within a percentage of its nominal value.

Constructing this specific ``UncertaintySet`` object can be done as follows:

.. code::

  >>> # === Define our pertinent data
  >>> percent_deviation = 0.15
  >>> bounds = [(nominal_values[i] - percent_deviation*nominal_values[i],
                 nominal_values[i] + percent_deviation*nominal_values[i])
                 for i in range(4)]

  >>> # === Construct the Box Set
  >>> box_uncertainty_set = pyros.BoxSet(bounds=bounds)


Step 3: Solve with PyROS
^^^^^^^^^^^^^^^^^^^^^^^^^^
The final step in solving a model with PyROS is to designate the remaining required inputs:
``first_stage_variables``, ``second_stage_variables``, ``uncertain_params``, ``local_solver``, and ``global_solver``.


The Single-Stage Problem
"""""""""""""""""""""""""
If we choose to designate all variables as either design or state variables without any control variables (i.e., all degrees of freedom are first-stage),
we can solve the single-stage problem as shown below.

.. code::

  >>> # === Designate which variables correspond to first- and second-stage degrees of freedom ===
  >>> first_stage_variables =[m.x1, m.x2, m.x3, m.x4, m.x5, m.x6,
                        m.x19, m.x20, m.x21, m.x22, m.x23, m.x24, m.x31]
  >>> second_stage_variables = []
  >>> # === The remaining variables are implicitly designated to be state variables

  >>> # === Specify which parameters are uncertain ===
  >>> uncertain_parameters = [m.p] # We can pass IndexedParams this way to PyROS, or as a expanded list per index

  >>> # === Designate local and global NLP solvers. Here we use BARON as both the local and the global NLP solver. ===
  >>> local_solver = pyo.SolverFactory('baron')
  >>> global_solver = pyo.SolverFactory('baron')

  >>> # === Make the PyROS solver object ===
  >>> pyros_solver = SolverFactory("pyros")

  >>> # === Solve the uncertain optimization problem ===
  >>> results = pyros_solver.solve(model = m,
                                   first_stage_variables = first_stage_variables,
                                   second_stage_variables = second_stage_variables,
                                   uncertain_params = uncertain_parameters,
                                   uncertainty_set = box_uncertainty_set,
                                   local_solver = local_solver
                                   global_solver= global_solver,
                                   options = {
                                       "objective_focus": pyros.ObjectiveType.worst_case,
                                       "solve_master_globally": True
                                    })

  >>> # === Print Results ===
  >>> const_dr_final_objective = pyo.value(pyros_results.final_objective_value)
  >>> print("\n Final objective value: %s" % const_dr_final_objective)

Final objective value: 48,349,406.72


The Two-Stage Problem
""""""""""""""""""""""
For this next set of runs,
we will assume that some of the previously designated first-stage degrees of freedom are in fact second-stage ones.

PyROS handles second-stage degrees of freedom via the use of decision rules.
Currently, PyROS supports constant, affine, and quadratic functionals to implement decision rules.
This is controlled with the config option ``decision_rule_order`` presented above.
For this next set of runs, we will assume we have a partition between first- and second-stage variables.

.. code::

  >>> # === Redefine the variable partitioning
  >>> first_stage_variables =[m.x5, m.x6, m.x19, m.x22, m.x23, m.x24, m.x31]
  >>> second_stage_variables = [m.x1, m.x2, m.x3, m.x4, m.x20, m.x21]

  >>> # === Call PyROS with decision_rule_order = 1 (affine decision rules)
  >>> affine_results = pyros_solver.solve(model = m,
                                          first_stage_variables = first_stage_variables,
                                          second_stage_variables = second_stage_variables,
                                          uncertain_params = uncertain_parameters,
                                          uncertainty_set = box_uncertainty_set,
                                          local_solver = local_solver,
                                          global_solver = global_solver,
                                          options = {
                                             "objective_focus": pyros.ObjectiveType.worst_case,
                                             "decision_rule_order": 1,
                                             "solve_master_globally": True
                                          })

  >>> # === Compare final objective to the constant decision rules case
  >>> affine_dr_objective = pyo.value(affine_results.final_objective_value)
  >>> percent_difference = (affine_dr_objective - const_dr_final_objective)/(const_dr_final_objective) * 100
  >>> print("\n Percent objective change relative to constant decision rules objective: %.2f %%" % percent_difference)

In this example, when we compare the final objective value in the case of constant decision rules (no second-stage recourse)
and affine decision rules, we see there is a -24.95% decrease in total objective value.

Price of Robustness
""""""""""""""""""""""
Using appropriately constructed hierarchies, PyROS allows for the facile comparison of robust optimal objectives across sets to determine the "price of robustness."
For the set we considered here, the ``BoxSet``, we can create such a hierarchy via an array of ``percent deviation`` parameters to define the size of these uncertainty sets.
We can then loop through the array of ``percent deviation`` values and call PyROS within a loop to identify robust solutions in light of each of the specified ``BoxSet`` objects.

.. code::

  >>> # === An array of % changes in the nominal uncertain parameter values to utilize in constructing BoxSets
  >>> percent_deviation_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

  >>> # === Final robust optimal objectives
  >>> robust_optimal_objectives = []

  >>> for percent_deviation in percent_deviation_list:
      ... bounds = [(nominal_values[i] - percent_deviation*nominal_values[i],
                     nominal_values[i] + percent_deviation*nominal_values[i])
                     for i in range(4)]
      ... box_uncertainty_set = pyros.BoxSet(bounds = bounds)
      ... results = pyros_solver.solve(model = m,
                                       first_stage_variables = first_stage_variables,
                                       second_stage_variables = second_stage_variables,
                                       uncertain_params = uncertain_parameters,
                                       uncertainty_set = box_uncertainty_set,
                                       local_solver = local_solver,
                                       global_solver = global_solver,
                                       options = {
                                          "objective_focus": pyros.ObjectiveType.worst_case,
                                          "decision_rule_order": 1,
                                          "solve_master_globally": True
                                       })
      ... robust_optimal_objectives.append(results.final_objective_value)

For this example, we obtain the following price of robustness results:

.. tabularcolumns:: |c|c|c|

+------------------------------------------+------------------------------+-----------------------------+
|  **Uncertainty Set Size (+/-)** :sup:`o` | **Robust Optimal Objective** | **% Increase** :sup:`x`     |
+------------------------------------------+------------------------------+-----------------------------+
|   0.00                                   | 35,837,659.18                | 0.00 %                      |
+------------------------------------------+------------------------------+-----------------------------+
|   0.05                                   | 35,986,085.98                | 0.41 %                      |
+------------------------------------------+------------------------------+-----------------------------+
|   0.10                                   | 36,135,191.59                | 0.82 %                      |
+------------------------------------------+------------------------------+-----------------------------+
|   0.15                                   | 36,285,254.72                | 1.23 %                      |
+------------------------------------------+------------------------------+-----------------------------+
|   0.20                                   | 36,437,979.81                | 1.64 %                      |
+------------------------------------------+------------------------------+-----------------------------+
|   0.25                                   | 37,946,802.00                | 5.55 %                      |
+------------------------------------------+------------------------------+-----------------------------+
|   0.30                                   | 43,478,190.92                | 17.57 %                     |
+------------------------------------------+------------------------------+-----------------------------+

..
    .. list-table::
       :widths: 10 10 10
       :header-rows: 1
       :align: center

       * - Uncertainty Set Size (+/-) :sup:`o`
         - Robust Optimal Objective
         - % Increase :sup:`x`
       * - 0.00
         - 35,837,659.18
         - 0.00 %
       * - 0.05
         - 35,986,085.98
         - 0.41 %
       * - 0.10
         - 36,135,191.59
         - 0.82 %
       * - 0.15
         - 36,285,254.72
         - 1.23 %
       * - 0.20
         - 36,437,979.81
         - 1.64 %
       * - 0.25
         - 37,946,802.00
         - 5.55 %

:sup:`o` **Relative Deviation around Nominal Realization**

:sup:`x` **Relative to Deterministic Optimal Objective**


This clearly illustrates the impact that the uncertainty set size can have on the robust optimal objective values.
Price of robustness studies like this are easily implemented using PyROS.

.. warning::

    PyROS is still under a beta release. Please provide feedback and/or report any issues by opening a Pyomo ticket.
