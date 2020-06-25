.. image:: /../logos/gdp/Pyomo-GDP-150.png
    :scale: 20%
    :class: no-scaled-link
    :align: right

************
Key Concepts
************

Generalized Disjunctive Programming (GDP) provides a way to bridge high-level propositional logic and algebraic constraints.
The GDP standard form from the :ref:`index page <gdp-main-page>` is repeated below.

.. math::

    \min\ obj = &\ f(x, z) \\
    \text{s.t.} \quad &\ Ax+Bz \leq d\\
    &\ g(x,z) \leq 0\\
    &\ \bigvee_{i\in D_k} \left[
        \begin{gathered}
        Y_{ik} \\
        M_{ik} x + N_{ik} z \leq e_{ik} \\
        r_{ik}(x,z)\leq 0\\
        \end{gathered}
    \right] \quad k \in K\\
    &\ \Omega(Y) = True \\
    &\ x \in X \subseteq \mathbb{R}^n\\
    &\ Y \in \{True, False\}^{p}\\
    &\ z \in Z \subseteq \mathbb{Z}^m

Original support in Pyomo.GDP focused on the disjuncts and disjunctions, allowing the modelers to group relational expressions in disjuncts, with disjunctions describing logical-OR relationships between the groupings.
As a result, we implemented the ``Disjunct`` and ``Disjunction`` objects before ``BooleanVar`` and the rest of the logical expression system.
Accordingly, we also describe the disjuncts and disjunctions first below.

Disjuncts
=========

Disjuncts represent groupings of relational expressions (e.g. algebraic constraints) summarized by a Boolean indicator variable :math:`Y` through implication:

.. math::

    \left.
    \begin{aligned}
    & Y_{ik} \Rightarrow & M_{ik} x + N_{ik} z &\leq e_{ik}\\
    & Y_{ik} \Rightarrow & r_{ik}(x,z) &\leq 0
    \end{aligned}
    \right.\qquad \forall i \in D_k, \forall k \in K


Logically, this means that if :math:`Y_{ik} = True`, then the constraints :math:`M_{ik} x + N_{ik} z \leq e_{ik}` and :math:`r_{ik}(x,z) \leq 0` must be satisfied.
However, if :math:`Y_{ik} = False`, then the corresponding constraints are ignored.
Note that :math:`Y_{ik} = False` does **not** imply that the corresponding constraints are *violated*.

.. _gdp-disjunctions-concept:

Disjunctions
============

Disjunctions describe a logical *OR* relationship between two or more Disjuncts.
The simplest and most common case is a 2-term disjunction:

.. math::

    \left[\begin{gathered}
    Y_1 \\
    \exp(x_2) - 1 = x_1 \\
    x_3 = x_4 = 0
    \end{gathered}
    \right] \bigvee \left[\begin{gathered}
    Y_2 \\
    \exp\left(\frac{x_4}{1.2}\right) - 1 = x_3 \\
    x_1 = x_2 = 0
    \end{gathered}
    \right]


The disjunction above describes the selection between two units in a process network.
:math:`Y_1` and :math:`Y_2` are the Boolean variables corresponding to the selection of process units 1 and 2, respectively.
The continuous variables :math:`x_1, x_2, x_3, x_4` describe flow in and out of the first and second units, respectively.
If a unit is selected, the nonlinear equality in the corresponding disjunct enforces the input/output relationship in the selected unit.
The final equality in each disjunct forces flows for the absent unit to zero.

Boolean Variables
=================

Boolean variables are decision variables that may take a value of ``True`` or ``False``.
These are most often encountered as the indicator variables of disjuncts.
However, they can also be independently defined to represent other problem decisions.

.. note::

    Boolean variables are not intended to participate in algebraic expressions.
    That is, :math:`3 \times \text{True}` does not make sense; hence, :math:`x = 3 Y_1` does not make sense.
    Instead, you may have the disjunction

    .. math::

        \left[\begin{gathered}
        Y_1 \\
        x = 3
        \end{gathered}
        \right] \bigvee \left[\begin{gathered}
        \neg Y_1 \\
        x = 0
        \end{gathered}
        \right]

Logical Propositions
====================

Logical propositions are constraints describing relationships between the Boolean variables in the model.

These logical propositions can include:

.. |neg| replace:: :math:`\neg Y_1`
.. |equiv| replace:: :math:`Y_1 \Leftrightarrow Y_2`
.. |land| replace:: :math:`Y_1 \land Y_2`
.. |lor| replace:: :math:`Y_1 \lor Y_2`
.. |xor| replace:: :math:`Y_1 \underline{\lor} Y_2`
.. |impl| replace:: :math:`Y_1 \Rightarrow Y_2`

+-----------------+---------+-------------+-------------+-------------+
| Operator        | Example | :math:`Y_1` | :math:`Y_2` | Result      |
+=================+=========+=============+=============+=============+
| Negation        | |neg|   | | ``True``  |             | | ``False`` |
|                 |         | | ``False`` |             | | ``True``  |
+-----------------+---------+-------------+-------------+-------------+
| Equivalence     | |equiv| | | ``True``  | | ``True``  | | ``True``  |
|                 |         | | ``True``  | | ``False`` | | ``False`` |
|                 |         | | ``False`` | | ``True``  | | ``False`` |
|                 |         | | ``False`` | | ``False`` | | ``True``  |
+-----------------+---------+-------------+-------------+-------------+
| Conjunction     | |land|  | | ``True``  | | ``True``  | | ``True``  |
|                 |         | | ``True``  | | ``False`` | | ``False`` |
|                 |         | | ``False`` | | ``True``  | | ``False`` |
|                 |         | | ``False`` | | ``False`` | | ``False`` |
+-----------------+---------+-------------+-------------+-------------+
| Disjunction     | |lor|   | | ``True``  | | ``True``  | | ``True``  |
|                 |         | | ``True``  | | ``False`` | | ``True``  |
|                 |         | | ``False`` | | ``True``  | | ``True``  |
|                 |         | | ``False`` | | ``False`` | | ``False`` |
+-----------------+---------+-------------+-------------+-------------+
| Exclusive OR    | |xor|   | | ``True``  | | ``True``  | | ``False`` |
|                 |         | | ``True``  | | ``False`` | | ``True``  |
|                 |         | | ``False`` | | ``True``  | | ``True``  |
|                 |         | | ``False`` | | ``False`` | | ``False`` |
+-----------------+---------+-------------+-------------+-------------+
| Implication     | |impl|  | | ``True``  | | ``True``  | | ``True``  |
|                 |         | | ``True``  | | ``False`` | | ``False`` |
|                 |         | | ``False`` | | ``True``  | | ``True``  |
|                 |         | | ``False`` | | ``False`` | | ``True``  |
+-----------------+---------+-------------+-------------+-------------+
