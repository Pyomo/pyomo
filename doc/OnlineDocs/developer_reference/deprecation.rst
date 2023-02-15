Deprecation and Removal of Functionality
========================================

During the course of development, there may be cases where it becomes
necessary to deprecate or remove functionality from the standard Pyomo
offering.

Deprecation
-----------

We offer a set of tools to help with deprecation in
``pyomo.common.deprecation``.

By policy, when deprecating or moving an existing capability,
one of the following functions should be imported. In use,
the ``version`` option should be set to ``TBD``, which
will be changed as of the next release.

.. autoclass:: pyomo.common.deprecation.deprecated

.. autoclass:: pyomo.common.deprecation.deprecation_warning

.. autoclass:: pyomo.common.deprecation.relocated_module

.. autoclass:: pyomo.common.deprecation.relocated_module_attribute

.. autoclass:: pyomo.common.deprecation.RenamedClass


Removal
-------

By policy, functionality should be deprecated with reasonable
warning, pending extenuating circumstances. The functionality should
be deprecatd, following the information above.

If the functionality is documented in the most recent
edition of [`Pyomo - Optimization in Python`_], it may not be removed
until the next major version release.

.. _Pyomo - Optimization in Python: https://doi.org/10.1007/978-3-030-68928-5

For other functionality, it is preferred that ample time is given
before removing the functionality. At minimum, significant functionality
removal will result in a minor version bump.
