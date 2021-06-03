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
    \displaystyle \text{min}_{x \in \mathcal{X}, z \in \mathbb{R}^n, y\in\mathbb{R}^a} & ~~ f_1\left(x\right) + f_2\left(x,z,y; q^0\right) & \\
    \displaystyle \text{s.t.} \quad \: & ~~ g_i\left(x, z, y; q^0\right) \leq 0 & \forall i \in \mathcal{I} \\
    & ~~ h_j\left(x,z,y; q^0\right) = 0 & \forall j \in \mathcal{J} \\
    \end{align*}

where:

* :math:`x \in \mathcal{X}` are the 'design' variables (i.e., first-stage degrees of freedom), where :math:`\mathcal{X} \subseteq \mathbb{R}^m` is the feasible space defined by the model constraints that only reference these variables
* :math:`z \in \mathbb{R}^n` are the 'control' variables (i.e., second-stage degrees of freedom)
* :math:`y \in \mathbb{R}^a` are the state variables
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
    & \ \displaystyle \min_{z \in \mathbb{R}^n, y \in \mathbb{R}^a} \ \ & \displaystyle ~~ f_1\left(x\right) + f_2\left(x, z, y, q\right) & & \\
    & & \text{s.t.} \quad \:& \displaystyle ~~ g_i\left(x, z, y, q\right) \leq 0 &  & \forall i \in \mathcal{I}\\
    & & & \displaystyle ~~ h_j\left(x, z, y, q\right) = 0 &  & \forall j \in \mathcal{J}
    \end{align*}

In order to solve problems of the above type, PyROS implements the Generalized Robust Cutting-Set algorithm developed in:

`Isenberg, NM, Akula, P, Eslick, JC, Bhattacharyya, D, Miller, DC, Gounaris, CE. A generalized cutting‐set approach for nonlinear robust optimization in process systems engineering. AIChE J. 2021; 67:e17175. DOI 10.1002/aic.17175 <https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.17175>`_

PyROS Required Inputs
-----------------------------
The required inputs to the PyROS solver are the following:

* The determinisitic optimization model
* The ``UncertaintySet`` object
* List of parameters to be considered uncertain
* List of first-stage variables
* List of second-stage variables
* Subordinate local and global optimization solvers

These arguments **must** be specified as ``kwargs`` when calling the ``solve`` command.

.. autoclass:: pyomo.contrib.pyros.pyros

deterministic_model : ``ConcreteModel``
    A ``ConcreteModel`` object representing the deterministic model.
uncertain_params : list(``Param``)
    A list of Pyomo ``Param`` objects in the ``deterministic_model`` to be considered uncertain. These specified ``Param`` objects must have the property ``mutable=True``.
uncertainty_set: ``UncertaintySet``
    A PyROS ``UncertaintySet`` object that represents the uncertainty in
Term
    Definition

I have no clue why the definition list below is classified as a different style
of definition list than the one above.

Is it the spaces in the term?
    Maybe it was the multiple line paragraph
    in the line below that caused this?

Is it the paragraph above the list maybe?
    I guess a lot of these lists don't have leading paragraphs?

Is it everything all at once?
    Who knows?!




PyROS Options
-----------------------------

PyROS UncertaintySets
-----------------------------


Example Problem Walk-through
-----------------------------

We will use a simple example to illustrate the usage of PyROS. The problem we will use is called *hydro* and comes from
the GAMS example problem database in `The GAMS Model Library <https://www.gams.com/latest/gamslib_ml/libhtml/>`_. The model was converted to Pyomo format via the `GAMS Convert tool <https://www.gams.com/latest/docs/S_CONVERT.html>`_.
Per our analysis, this model is a QCQP with 13 decision variables, 18 state variables, and we have selected 3 uncertain parameters.

In this model, are 6 linear inequality constraints, 6 linear equality constraints, 6 non-linear (quadratic) equalities, and a quadratic objective.

Note that we can easily verify that this model satisfies our assumption regarding 1-1 state variable mappings via an elimination-type algorithm for variables and equality constraints.

Step 1: Define the Deterministic Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Step 2: Define the Uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Step 3: Solve with PyROS
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Single-Stage Problem
"""""""""""""""""""""""""

The Two-Stage Problem
""""""""""""""""""""""

Price of Robustness
""""""""""""""""""""""

.. warning::

   PyROS is currently a beta code and is subject to change.
