.. _pyros_uncertainty_sets:

======================
PyROS Uncertainty Sets
======================

.. contents:: Table of Contents
   :depth: 1
   :local:


Overview
========
In PyROS, the uncertainty set of a robust optimization problem
is represented by an instance of a subclass of the
:class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`
abstract base class.
PyROS provides a suite of
:ref:`pre-implemented concrete subclasses <pyros_pre_implemented_types>`
to facilitate instantiation of uncertainty sets
that are commonly used in the optimization literature.
:ref:`Custom uncertainty set types <pyros_custom_sets>`
can be implemented by subclassing
:class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`.


.. note::
   The :class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`
   class is an abstract class and therefore cannot be
   directly instantiated.


.. _pyros_pre_implemented_types:

Pre-Implemented Uncertainty Set Types
=====================================
The pre-implemented 
:class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`
subclasses are enumerated below:

.. autosummary::

   ~pyomo.contrib.pyros.uncertainty_sets.AxisAlignedEllipsoidalSet
   ~pyomo.contrib.pyros.uncertainty_sets.BoxSet
   ~pyomo.contrib.pyros.uncertainty_sets.BudgetSet
   ~pyomo.contrib.pyros.uncertainty_sets.CardinalitySet
   ~pyomo.contrib.pyros.uncertainty_sets.DiscreteScenarioSet
   ~pyomo.contrib.pyros.uncertainty_sets.EllipsoidalSet
   ~pyomo.contrib.pyros.uncertainty_sets.FactorModelSet
   ~pyomo.contrib.pyros.uncertainty_sets.IntersectionSet
   ~pyomo.contrib.pyros.uncertainty_sets.PolyhedralSet


Mathematical definitions of the pre-implemented 
:class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`
subclasses are provided below.

.. _pyros_uncertainty_sets_math_defs:
  
.. list-table:: Mathematical definitions of PyROS uncertainty sets of dimension :math:`n`.
   :header-rows: 1
   :class: scrollwide-table

   * - Uncertainty Set Type
     - Input Data
     - Mathematical Definition
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.AxisAlignedEllipsoidalSet`
     - :math:`\begin{array}{l} q^0 \in \mathbb{R}^{n}, \\ \alpha \in \mathbb{R}_{+}^{n} \end{array}`
     - :math:`\left\{ q \in \mathbb{R}^{n} \middle| \begin{array}{l} \displaystyle\sum_{\substack{i = 1: \\ \alpha_{i} > 0}}^{n}  \left(\frac{q_{i} - q_{i}^{0}}{\alpha_{i}}\right)^2 \leq 1 \\ q_{i} = q_{i}^{0} \,\forall\,i : \alpha_{i} = 0 \end{array} \right\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.BoxSet`
     - :math:`\begin{array}{l} q ^{\text{L}} \in \mathbb{R}^{n}, \\ q^{\text{U}} \in \mathbb{R}^{n} \end{array}`
     - :math:`\{q \in \mathbb{R}^n \mid q^\mathrm{L} \leq q \leq q^\mathrm{U}\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.BudgetSet`
     - :math:`\begin{array}{l} q^{0} \in \mathbb{R}^{n}, \\ b \in \mathbb{R}_{+}^{L}, \\ B \in \{0, 1\}^{L \times n} \end{array}`
     - :math:`\left\{ q \in \mathbb{R}^{n} \middle| \begin{array}{l} \begin{pmatrix} B \\ -I \end{pmatrix} q \leq \begin{pmatrix}  b + Bq^{0} \\ -q^{0} \end{pmatrix}  \end{array} \right\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.CardinalitySet`
     - :math:`\begin{array}{l} q^{0} \in \mathbb{R}^{n}, \\ \hat{q} \in \mathbb{R}_{+}^{n}, \\ \Gamma \in [0, n] \end{array}`
     - :math:`\left\{ q \in \mathbb{R}^{n} \middle| \begin{array}{l} \exists\,\xi \in [0, 1]^n\,:\\ \quad \,q = q^{0} + \hat{q} \circ \xi \\ \quad \displaystyle \sum_{i=1}^{n} \xi_{i} \leq \Gamma \end{array} \right\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.DiscreteScenarioSet`
     - :math:`q^{1}, q^{2},\dots , q^{S} \in \mathbb{R}^{n}`
     - :math:`\{q^{1}, q^{2}, \dots , q^{S}\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.EllipsoidalSet`
     - :math:`\begin{array}{l} q^0 \in \mathbb{R}^n, \\ P \in \mathbb{S}_{++}^{n}, \\ s \in \mathbb{R}_{+} \end{array}`
     - :math:`\{q \in \mathbb{R}^{n} \mid (q - q^{0})^{\intercal} P^{-1} (q - q^{0}) \leq s\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.FactorModelSet`
     - :math:`\begin{array}{l} q^{0} \in \mathbb{R}^{n}, \\ \Psi \in \mathbb{R}^{n \times F}, \\ \beta \in [0, 1] \end{array}`
     - :math:`\left\{ q \in \mathbb{R}^{n} \middle| \begin{array}{l} \exists\,\xi \in [-1, 1]^F \,:\\ \quad q = q^{0} + \Psi \xi \\ \quad \displaystyle\bigg| \sum_{j=1}^{F} \xi_{j} \bigg| \leq \beta F \end{array} \right\}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.IntersectionSet`
     - :math:`\mathcal{Q}_{1}, \mathcal{Q}_{2}, \dots , \mathcal{Q}_{m} \subset \mathbb{R}^{n}`
     - :math:`\displaystyle \bigcap_{i=1}^{m} \mathcal{Q}_{i}`
   * - :class:`~pyomo.contrib.pyros.uncertainty_sets.PolyhedralSet`
     - :math:`\begin{array}{l} A \in \mathbb{R}^{m \times n}, \\ b \in \mathbb{R}^{m}\end{array}`
     - :math:`\{q \in \mathbb{R}^{n} \mid A q \leq b\}`


.. _pyros_custom_sets:

Custom Uncertainty Set Types
============================
A custom uncertainty set type
in which all uncertain parameters
are modeled as continuous quantities
can be implemented by subclassing
:class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`.
For discrete sets, we recommend using the pre-implemented
:class:`~pyomo.contrib.pyros.uncertainty_sets.DiscreteScenarioSet`
subclass instead of implementing a new set type.
PyROS does not support mixed-integer uncertainty set types.
