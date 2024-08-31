pyomo.common.config
===================

.. currentmodule:: pyomo.common.config

Core classes
~~~~~~~~~~~~

.. autosummary::

   ConfigDict
   ConfigList
   ConfigValue

Utilities
~~~~~~~~~

.. autosummary::

   document_kwargs_from_configdict


Domain validators
~~~~~~~~~~~~~~~~~

.. autosummary::

   Bool
   Integer
   PositiveInt
   NegativeInt
   NonNegativeInt
   NonPositiveInt
   PositiveFloat
   NegativeFloat
   NonPositiveFloat
   NonNegativeFloat
   In
   IsInstance
   InEnum
   ListOf
   Module
   Path
   PathList
   DynamicImplicitDomain

.. autoclass:: ConfigBase
   :members:
   :undoc-members:

.. autoclass:: ConfigDict
   :show-inheritance:
   :members:
   :undoc-members:

.. autoclass:: ConfigList
   :show-inheritance:
   :members:
   :undoc-members:

.. autoclass:: ConfigValue
   :show-inheritance:
   :members:
   :undoc-members:

.. autodecorator:: document_kwargs_from_configdict

.. autofunction:: Bool
.. autofunction:: Integer
.. autofunction:: PositiveInt
.. autofunction:: NegativeInt
.. autofunction:: NonNegativeInt
.. autofunction:: NonPositiveInt
.. autofunction:: PositiveFloat
.. autofunction:: NegativeFloat
.. autofunction:: NonPositiveFloat
.. autofunction:: NonNegativeFloat
.. autoclass:: In
.. autoclass:: IsInstance
.. autoclass:: InEnum
.. autoclass:: ListOf
.. autoclass:: Module
.. autoclass:: Path
.. autoclass:: PathList
.. autoclass:: DynamicImplicitDomain
