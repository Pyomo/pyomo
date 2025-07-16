.. role:: python(code)
   :language: python

.. warning::

   The :python:`pyomo.kernel` API is still in the beta phase of development. It is fully tested and functional; however, the interface may change as it becomes further integrated with the rest of Pyomo.

.. warning::

   Models built with :python:`pyomo.kernel` components are not yet compatible with pyomo extension modules (e.g., :python:`PySP`, :python:`pyomo.dae`, :python:`pyomo.gdp`).

The Kernel Library API Reference
================================

.. _kernel_modeling_components:

Modeling Components:
^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   block.rst
   variable.rst
   constraint.rst
   parameter.rst
   objective.rst
   expression.rst
   sos.rst
   suffix.rst
   piecewise/index.rst
   conic.rst

Base API:
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   base.rst
   homogeneous_container.rst
   heterogeneous_container.rst

Containers:
^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   tuple_container.rst
   list_container.rst
   dict_container.rst
