Pyomo Nonlinear Preprocessing
=============================

``pyomo.contrib.preprocessing`` is a contributed library of preprocessing
transformations intended to operate upon nonlinear and mixed-integer nonlinear
programs (NLPs and MINLPs), as well as generalized disjunctive programs (GDPs).

This contributed package is maintained by `Qi Chen
<https://github.com/qtothec>`_ and `his colleagues from Carnegie Mellon
University <http://capd.cheme.cmu.edu/>`_.

The following preprocessing transformations are available. However, some may
later be deprecated or combined, depending on their usefulness.

.. currentmodule:: pyomo.contrib.preprocessing.plugins

.. autosummary::
    :nosignatures:

    var_aggregator.VariableAggregator
    bounds_to_vars.ConstraintToVarBoundTransform
    induced_linearity.InducedLinearity
    constraint_tightener.TightenContraintFromVars
    deactivate_trivial_constraints.TrivialConstraintDeactivator
    detect_fixed_vars.FixedVarDetector
    equality_propagate.FixedVarPropagator
    equality_propagate.VarBoundPropagator
    init_vars.InitMidpoint
    init_vars.InitZero
    remove_zero_terms.RemoveZeroTerms
    strip_bounds.VariableBoundStripper
    zero_sum_propagator.ZeroSumPropagator


Variable Aggregator
-------------------

The following code snippet demonstrates usage of the variable aggregation
transformation on a concrete Pyomo model:

.. doctest::

    >>> from pyomo.environ import *
    >>> m = ConcreteModel()
    >>> m.v1 = Var(initialize=1, bounds=(1, 8))
    >>> m.v2 = Var(initialize=2, bounds=(0, 3))
    >>> m.v3 = Var(initialize=3, bounds=(-7, 4))
    >>> m.v4 = Var(initialize=4, bounds=(2, 6))
    >>> m.c1 = Constraint(expr=m.v1 == m.v2)
    >>> m.c2 = Constraint(expr=m.v2 == m.v3)
    >>> m.c3 = Constraint(expr=m.v3 == m.v4)
    >>> TransformationFactory('contrib.aggregate_vars').apply_to(m)

To see the results of the transformation, you could then use the command

.. code::

    >>> m.pprint()

.. autoclass:: pyomo.contrib.preprocessing.plugins.var_aggregator.VariableAggregator
    :members: apply_to, create_using, update_variables


Explicit Constraints to Variable Bounds
---------------------------------------

.. doctest::

    >>> from pyomo.environ import *
    >>> m = ConcreteModel()
    >>> m.v1 = Var(initialize=1)
    >>> m.v2 = Var(initialize=2)
    >>> m.v3 = Var(initialize=3)
    >>> m.c1 = Constraint(expr=m.v1 == 2)
    >>> m.c2 = Constraint(expr=m.v2 >= -2)
    >>> m.c3 = Constraint(expr=m.v3 <= 5)
    >>> TransformationFactory('contrib.constraints_to_var_bounds').apply_to(m)

.. autoclass:: pyomo.contrib.preprocessing.plugins.bounds_to_vars.ConstraintToVarBoundTransform
    :members: apply_to, create_using


Induced Linearity Reformulation
-------------------------------

.. autoclass:: pyomo.contrib.preprocessing.plugins.induced_linearity.InducedLinearity
    :members: apply_to, create_using


Constraint Bounds Tightener
---------------------------

This transformation was developed by `Sunjeev Kale
<https://github.com/sjkale>`_ at Carnegie Mellon University.

.. autoclass:: pyomo.contrib.preprocessing.plugins.constraint_tightener.TightenContraintFromVars
    :members: apply_to, create_using

Trivial Constraint Deactivation
-------------------------------

.. autoclass:: pyomo.contrib.preprocessing.plugins.deactivate_trivial_constraints.TrivialConstraintDeactivator
    :members: apply_to, create_using, revert

Fixed Variable Detection
------------------------

.. autoclass:: pyomo.contrib.preprocessing.plugins.detect_fixed_vars.FixedVarDetector
    :members: apply_to, create_using, revert

Fixed Variable Equality Propagator
----------------------------------

.. autoclass:: pyomo.contrib.preprocessing.plugins.equality_propagate.FixedVarPropagator
    :members: apply_to, create_using, revert

Variable Bound Equality Propagator
----------------------------------

.. autoclass:: pyomo.contrib.preprocessing.plugins.equality_propagate.VarBoundPropagator
    :members: apply_to, create_using, revert

Variable Midpoint Initializer
-----------------------------

.. autoclass:: pyomo.contrib.preprocessing.plugins.init_vars.InitMidpoint
    :members: apply_to, create_using

Variable Zero Initializer
-------------------------

.. autoclass:: pyomo.contrib.preprocessing.plugins.init_vars.InitZero
    :members: apply_to, create_using

Zero Term Remover
-----------------

.. autoclass:: pyomo.contrib.preprocessing.plugins.remove_zero_terms.RemoveZeroTerms
    :members: apply_to, create_using

Variable Bound Remover
----------------------

.. autoclass:: pyomo.contrib.preprocessing.plugins.strip_bounds.VariableBoundStripper
    :members: apply_to, create_using, revert

Zero Sum Propagator
-------------------

.. autoclass:: pyomo.contrib.preprocessing.plugins.zero_sum_propagator.ZeroSumPropagator
    :members: apply_to, create_using
