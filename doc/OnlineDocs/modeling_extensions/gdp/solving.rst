.. image:: /../logos/gdp/Pyomo-GDP-150.png
    :scale: 20%
    :class: no-scaled-link
    :align: right

*****************************************
Solving Logic-based Models with Pyomo.GDP
*****************************************


Flexible Solution Suite
=======================

Once a model is formulated as a GDP model, a range of solution
strategies are available to manipulate and solve it.

The traditional approach is reformulation to a MI(N)LP, but various
other techniques are possible, including direct solution via the
:ref:`GDPopt solver <gdpopt-main-page>`.  Below, we describe some of
these capabilities.

.. _gdp-reformulations:

Reformulations
==============

Logical constraints
-------------------

.. note::

    Historically users needed to explicitly convert logical propositions
    to algebraic form prior to invoking the GDP MI(N)LP reformulations
    or the GDPopt solver. However, this is mathematically incorrect
    since the GDP MI(N)LP reformulations themselves convert logical
    formulations to algebraic formulations.  The current recommended
    practice is to pass the entire (mixed logical / algebraic) model to
    the MI(N)LP reformulations or GDPopt directly.

There are several approaches to convert logical constraints into
algebraic form.

Conjunctive Normal Form
^^^^^^^^^^^^^^^^^^^^^^^

The first transformation (`core.logical_to_linear`) leverages the
`sympy` package to generate the conjunctive normal form of the logical
constraints and then adds the equivalent as a list algebraic
constraints.  The following transforms logical propositions on the model
to algebraic form:

.. code::

    TransformationFactory('core.logical_to_linear').apply_to(model)

The transformation creates a constraint list with a unique name starting
with ``logic_to_linear``, within which the algebraic equivalents of the
logical constraints are placed.  If not already associated with a binary
variable, each ``BooleanVar`` object will receive a generated binary
counterpart.  These associated binary variables may be accessed via the
``get_associated_binary()`` method.

.. code::

    m.Y[1].get_associated_binary()

Additional augmented variables and their corresponding constraints may
also be created, as described in :ref:`gdp-advanced-examples`.

Following solution of the GDP model, values of the Boolean variables may be updated from their algebraic binary counterparts using the ``update_boolean_vars_from_binary()`` function.

.. autofunction:: pyomo.core.plugins.transform.logical_to_linear.update_boolean_vars_from_binary

Factorable Programming
^^^^^^^^^^^^^^^^^^^^^^

The second transformation (`contrib.logical_to_disjunctive`) leverages
ideas from factorable programming to first generate an equivalent set of
"factored" logical constraints form by traversing each logical
proposition and replacing each logical operator with an additional
Boolean variable and then adding the "simple" logical constraint that
equates the new Boolean variable with the single logical operator.

The resulting "simple" logical constraints are converted to either MIP
or GDP form: if the constraint contains only Boolean variables, then
then MIP representation is emitted.  Logical constraints with mixed
integer-Boolean arguments (e.g., `atmost`, `atleast`, `exactly`, etc.)
are converted to a disjunctive representation.

As this transformation both avoids the conversion into `sympy` and only
requires a single traversal of each logical constraint,
`contrib.logical_to_disjunctive` is significantly faster than
`core.logical_to_linear` at the cost of a larger model.  In practice,
the cost of the larger model is negated by the effectiveness of the MIP
presolve in most solvers.

Reformulation to MI(N)LP
------------------------

To use standard commercial solvers, you must convert the disjunctive
model to a standard MILP/MINLP model.  The two classical strategies for
doing so are the (included) Big-M and Hull reformulations.


Big-M (BM) Reformulation
^^^^^^^^^^^^^^^^^^^^^^^^

The Big-M reformulation\ [#gdp-bm]_ results in a smaller transformed model, avoiding the need to add extra variables; however, it yields a looser continuous relaxation.
By default, the BM transformation will estimate reasonably tight M values for you if variables are bounded.
For nonlinear models where finite expression bounds may be inferred from variable bounds, the BM transformation may also be able to automatically compute M values for you.
For all other models, you will need to provide the M values through a "BigM" Suffix, or through the `bigM` argument to the transformation.
We will raise a ``GDP_Error`` for missing M values.

To apply the BM reformulation within a python script, use:

.. code::

    TransformationFactory('gdp.bigm').apply_to(model)

From the Pyomo command line, include the ``--transform pyomo.gdp.bigm`` option.

Multiple Big-M (MBM) Reformulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We also implement the multiple-parameter Big-M (MBM) approach described in literature\ [#gdp-mbm]_.
By default, the MBM transformation will solve continuous subproblems in order to calculate M values.
This process can be time-consuming, so the transformation also provides a method to export the M values used as a dictionary and allows for M values to be provided through the `bigM` argument.

For example, to apply the transformation and store the M values, use:

.. code::

    mbigm = TransformationFactory('gdp.mbigm')
    mbigm.apply_to(model)

    # These can be stored...
    M_values = mbigm.get_all_M_values(model)
    # ...so that in future runs, you can write:
    mbigm.apply_to(m, bigM=M_values)

From the Pyomo command line, include the ``--transform pyomo.gdp.mbigm`` option.

.. warning::
   The Multiple Big-M transformation does not currently support Suffixes and will
   ignore "BigM" Suffixes.

Hull Reformulation (HR)
^^^^^^^^^^^^^^^^^^^^^^^

The Hull Reformulation requires a lifting into a higher-dimensional space and consequently introduces disaggregated variables and their corresponding constraints.

.. note::

    - All variables that appear in disjuncts need upper and lower bounds.

    - The hull reformulation is an exact reformulation at the solution
      points even for nonconvex GDP models, but the resulting MINLP will
      also be nonconvex.

To apply the Hull reformulation within a python script, use:

.. code::

    TransformationFactory('gdp.hull').apply_to(model)

From the Pyomo command line, include the ``--transform pyomo.gdp.hull`` option.

Hybrid BM/HR Reformulation
^^^^^^^^^^^^^^^^^^^^^^^^^^

An experimental (for now) implementation of the cutting plane approach described in literature\ [#gdp-cuttingplanes]_ is provided for linear GDP models.
The transformation augments the BM reformulation by a set of cutting planes generated from the HR model by solving separation problems.
This gives a model that is not as large as the HR, but with a stronger continuous relaxation than the BM.

This transformation is accessible via:

.. code::

    TransformationFactory('gdp.cuttingplane').apply_to(model)

Direct GDP solvers
==================

Pyomo includes the contributed GDPopt solver, which can directly solve
GDP models.  Its usage is described within the :ref:`contributed
packages documentation <gdpopt-main-page>`.

References
==========

.. [#gdp-pse-paper] Chen, Q., Johnson, E. S., Siirola, J. D., & Grossmann, I. E. (2018). Pyomo.GDP: Disjunctive Models in Python. In M. R. Eden, M. G. Ierapetritou, & G. P. Towler (Eds.), *Proceedings of the 13th International Symposium on Process Systems Engineering* (pp. 889–894). San Diego: Elsevier B.V. https://doi.org/10.1016/B978-0-444-64241-7.50143-9

.. [#gdp-main-paper] Chen, Q., Johnson, E. S., Bernal, D. E., Valentin, R., Kale, S., Bates, J., Siirola, J. D. and Grossmann, I. E. (2021). Pyomo.GDP: an ecosystem for logic based modeling and optimization development, *Optimization and Engineering* (pp. 1-36).https://doi.org/10.1007/s11081-021-09601-7 

.. [#gdp-review-2013] Grossmann, I. E., & Trespalacios, F. (2013). Systematic modeling of discrete-continuous optimization models through generalized disjunctive programming. *AIChE Journal*, 59(9), 3276–3295. https://doi.org/10.1002/aic.14088

.. [#gdp-mbm] Trespalacios, F., & Grossmann, I. E. (2015). Improved Big-M reformulation for generalized disjunctive programs. *Computers and Chemical Engineering*, 76, 98–103. https://doi.org/10.1016/j.compchemeng.2015.02.013

.. [#gdp-bm] Nemhauser, G. L., & Wolsey, L. A. (1988). *Integer and combinatorial optimization*. New York: Wiley.

.. [#gdp-cuttingplanes] Sawaya, N. W., & Grossmann, I. E. (2003). A cutting plane method for solving linear generalized disjunctive programming problems. *Computer Aided Chemical Engineering*, 15(C), 1032–1037. https://doi.org/10.1016/S1570-7946(03)80444-3
