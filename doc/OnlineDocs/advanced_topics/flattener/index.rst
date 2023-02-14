"Flattening" a Pyomo model
==========================

.. autosummary::

   pyomo.dae.flatten

.. toctree::
   :maxdepth: 1

   motivation.rst
   reference.rst

What does it mean to flatten a model?
-------------------------------------
When accessing components in a block-structured model, we use
``component_objects`` or ``component_data_objects`` to access all objects
of a specific ``Component`` or ``ComponentData`` type.
The generated objects may be thought of as a "flattened" representation
of the model, as they may be accessed without any knowledge of the model's
block structure.
These methods are very useful, but it is still challenging to use them
to access specific components.
Specifically, we often want to access "all components indexed by some set,"
or "all component data at a particular index of this set."
In addition, we often want to generate the components in a block that
is indexed by our particular set, as these components may be thought of as
"implicitly indexed" by this set.
The ``pyomo.dae.flatten`` module aims to address this use case by providing
utilities to generate all components indexed, explicitly or implicitly, by
user-provided sets.

**When we say "flatten a model," we mean "generate all components in the model,
preserving all user-specified indexing sets."**

Data structures
---------------
The components returned are either ``ComponentData`` objects, for components
not indexed by any of the provided sets, or references-to-slices, for
components indexed, explicitly or implicitly, by the provided sets.
Slices are necessary as they can encode "implicit indexing" -- where a
component is contained in an indexed block. It is natural to return references
to these slices, so they may be accessed and manipulated like any other
component.
