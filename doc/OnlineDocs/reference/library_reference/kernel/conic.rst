Conic Constraints
=================

A collection of classes that provide an easy and performant
way to declare conic constraints. The Mosek solver interface
includes special handling of these objects that recognizes
them as convex constraints. Other solver interfaces will
treat these objects as general nonlinear or quadratic
expressions, and may or may not have the ability to identify
their convexity.

Summary
~~~~~~~
.. autosummary::

   pyomo.core.kernel.conic.quadratic
   pyomo.core.kernel.conic.rotated_quadratic
   pyomo.core.kernel.conic.primal_exponential
   pyomo.core.kernel.conic.primal_power
   pyomo.core.kernel.conic.dual_exponential
   pyomo.core.kernel.conic.dual_power

Member Documentation
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: pyomo.core.kernel.conic.quadratic
   :show-inheritance:
   :members:
.. autoclass:: pyomo.core.kernel.conic.rotated_quadratic
   :show-inheritance:
   :members:
.. autoclass:: pyomo.core.kernel.conic.primal_exponential
   :show-inheritance:
   :members:
.. autoclass:: pyomo.core.kernel.conic.primal_power
   :show-inheritance:
   :members:
.. autoclass:: pyomo.core.kernel.conic.dual_exponential
   :show-inheritance:
   :members:
.. autoclass:: pyomo.core.kernel.conic.dual_power
   :show-inheritance:
   :members:
