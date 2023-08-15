Deprecation and Removal of Functionality
========================================

During the course of development, there may be cases where it becomes
necessary to deprecate or remove functionality from the standard Pyomo
offering.

Deprecation
-----------

We offer a set of tools to help with deprecation in
``pyomo.common.deprecation``.

By policy, when deprecating or moving an existing capability, one of the
following utilities should be leveraged.  Each has a required
``version`` argument that should be set to current development version (e.g.,
``"6.6.2.dev0"``).  This version will be updated to the next actual
release as part of the Pyomo release process.  The current development version
can be found by running ``pyomo --version`` on your local fork/branch.

.. currentmodule:: pyomo.common.deprecation

.. autosummary::

   deprecated
   deprecation_warning
   relocated_module
   relocated_module_attribute
   RenamedClass

.. autodecorator:: pyomo.common.deprecation.deprecated
   :noindex:

.. autofunction:: pyomo.common.deprecation.deprecation_warning
   :noindex:

.. autofunction:: pyomo.common.deprecation.relocated_module
   :noindex:

.. autofunction:: pyomo.common.deprecation.relocated_module_attribute
   :noindex:

.. autoclass:: pyomo.common.deprecation.RenamedClass
   :noindex:


Removal
-------

By policy, functionality should be deprecated with reasonable
warning, pending extenuating circumstances. The functionality should
be deprecated, following the information above.

If the functionality is documented in the most recent
edition of [`Pyomo - Optimization Modeling in Python`_], it may not be removed
until the next major version release.

.. _Pyomo - Optimization Modeling in Python: https://doi.org/10.1007/978-3-030-68928-5

For other functionality, it is preferred that ample time is given
before removing the functionality. At minimum, significant functionality
removal will result in a minor version bump.
