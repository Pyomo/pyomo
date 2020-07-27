.. _gdp-main-page:

***********************************
Generalized Disjunctive Programming
***********************************

.. image:: /../logos/gdp/Pyomo-GDP-150.png
   :scale: 35%
   :align: right
   :class: no-scaled-link

The Pyomo.GDP modeling extension provides support for Generalized Disjunctive Programming (GDP)\ [#gdp]_, an extension of Disjunctive Programming\ [#dp]_ from the operations research community to include nonlinear relationships. The classic form for a GDP is given by:

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

Here, we have the minimization of an objective :math:`obj` subject to global linear constraints :math:`Ax+Bz \leq d` and nonlinear constraints :math:`g(x,z) \leq 0`, with conditional linear constraints :math:`M_{ik} x + N_{ik} z \leq e_{ik}` and nonlinear constraints :math:`r_{ik}(x,z)\leq 0`.
These conditional constraints are collected into disjuncts :math:`D_k`, organized into disjunctions :math:`K`. Finally, there are logical propositions :math:`\Omega(Y) = True`.
Decision/state variables can be continuous :math:`x`, Boolean :math:`Y`, and/or integer :math:`z`.

GDP is useful to model discrete decisions that have implications on the system behavior\ [#gdpreview]_.
For example, in process design, a disjunction may model the choice between processes A and B.
If A is selected, then its associated equations and inequalities will apply; otherwise, if B is selected, then its respective constraints should be enforced.

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
    Y_1 \underline{\vee} Y_2
    \end{gather*}

Here, if the Boolean :math:`Y_1` is ``True``, then the constraints in the first disjunct are enforced; otherwise, the constraints in the second disjunct are enforced.
The following sections describe the key concepts, modeling, and solution approaches available for Generalized Disjunctive Programming.

.. toctree::
    :caption: Pyomo.GDP Contents
    :maxdepth: 2

    concepts
    modeling
    solving

Literature References
=====================

.. [#gdp] Raman, R., & Grossmann, I. E. (1994). Modelling and computational techniques for logic based integer programming. *Computers & Chemical Engineering*, 18(7), 563–578. https://doi.org/10.1016/0098-1354(93)E0010-7

.. [#dp] Balas, E. (1985). Disjunctive Programming and a Hierarchy of Relaxations for Discrete Optimization Problems. *SIAM Journal on Algebraic Discrete Methods*, 6(3), 466–486. https://doi.org/10.1137/0606047

.. [#gdpreview] Grossmann, I. E., & Trespalacios, F. (2013). Systematic modeling of discrete-continuous optimization models through generalized disjunctive programming. *AIChE Journal*, 59(9), 3276–3295. https://doi.org/10.1002/aic.14088
