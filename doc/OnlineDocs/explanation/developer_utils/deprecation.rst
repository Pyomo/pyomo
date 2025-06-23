Deprecation and Removal of Functionality
========================================

During the course of development, there may be cases where it becomes
necessary to deprecate or remove functionality from the standard Pyomo
offering.

Deprecation
-----------

We offer a set of tools to help with deprecation in
:py:mod:`pyomo.common.deprecation`.

By policy, when deprecating or moving an existing capability, one of the
following utilities should be leveraged.  Each has a required
``version`` argument that should be set to current development version (e.g.,
``"6.6.2.dev0"``).  This version will be updated to the next actual
release as part of the Pyomo release process.  The current development version
can be found by running

   ``pyomo --version``

on your local fork/branch.

.. currentmodule:: pyomo.common.deprecation

.. autosummary::

   deprecated
   deprecation_warning
   moved_module
   relocated_module_attribute
   RenamedClass


Removal
-------

By policy, functionality should be deprecated with reasonable
warning, pending extenuating circumstances. The functionality should
be deprecated, following the information above.

If the functionality is documented in the most recent
edition of :ref:`Pyomo - Optimization Modeling in Python <pyomobookiii>`,
it may not be removed until the next major version release.

For other functionality, it is preferred that ample time is given
before removing the functionality. At minimum, significant functionality
removal will result in a minor version bump.
