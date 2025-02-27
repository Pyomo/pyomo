.. _gdp-main-page:

***********************************
Generalized Disjunctive Programming
***********************************

.. image:: /../logos/gdp/Pyomo-GDP-150.png
   :scale: 35%
   :align: right
   :class: no-scaled-link

The Pyomo.GDP modeling extension [PyomoGDP-proceedings]_
[PyomoGDP-paper]_ provides support for Generalized Disjunctive
Programming (GDP) [RG94]_, an extension of Disjunctive Programming
[Bal85]_ from the operations research community to include nonlinear
relationships. The classic form for a GDP is given by:

.. math::
   :nowrap:

   \[\begin{array}{ll}
    \min & f(x, z) \\
    \mathrm{s.t.} \quad & Ax+Bz \leq d\\
    & g(x,z) \leq 0\\
    & \bigvee_{i\in D_k} \left[
        \begin{gathered}
        Y_{ik} \\
        M_{ik} x + N_{ik} z \leq e_{ik} \\
        r_{ik}(x,z)\leq 0\\
        \end{gathered}
    \right] \quad k \in K\\
    & \Omega(Y) = True \\
    & x \in X \subseteq \mathbb{R}^n\\
    & Y \in \{True, False\}^{p}\\
    & z \in Z \subseteq \mathbb{Z}^m
   \end{array}\]

Here, we have the minimization of an objective :math:`f(x, z)` subject to global linear constraints :math:`Ax+Bz \leq d` and nonlinear constraints :math:`g(x,z) \leq 0`, with conditional linear constraints :math:`M_{ik} x + N_{ik} z \leq e_{ik}` and nonlinear constraints :math:`r_{ik}(x,z)\leq 0`.
These conditional constraints are collected into disjuncts :math:`D_k`, organized into disjunctions :math:`K`. Finally, there are logical propositions :math:`\Omega(Y) = True`.
Decision/state variables can be continuous :math:`x`, Boolean :math:`Y`, and/or integer :math:`z`.

GDP is useful to model discrete decisions that have implications on the
system behavior [GT13]_.  For example, in process design, a
disjunction may model the choice between processes A and B.  If A is
selected, then its associated equations and inequalities will apply;
otherwise, if B is selected, then its respective constraints should be
enforced.

Modelers often ask to model if-then-else relationships.
These can be expressed as a disjunction as follows:

.. math::
    :nowrap:

    \begin{gather*}
    \left[\begin{gathered}
    Y_1 \\
    \text{constraints} \\
    \text{for }\textit{then}
    \end{gathered}\right]
    \vee
    \left[\begin{gathered}
    Y_2 \\
    \text{constraints} \\
    \text{for }\textit{else}
    \end{gathered}\right] \\
    Y_1 \veebar Y_2
    \end{gather*}

Here, if the Boolean :math:`Y_1` is ``True``, then the constraints in the first disjunct are enforced; otherwise, the constraints in the second disjunct are enforced.
The following sections describe the key concepts, modeling, and solution approaches available for Generalized Disjunctive Programming.

.. toctree::
    :caption: Pyomo.GDP Contents
    :maxdepth: 2

    concepts
    modeling
    solving

