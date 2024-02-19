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

**When we say "flatten a model," we mean "recursively generate all components in
the model," where a component can be indexed only by user-specified indexing
sets (or is not indexed at all)**.

Data structures
---------------
The components returned are either ``ComponentData`` objects, for components
not indexed by any of the provided sets, or references-to-slices, for
components indexed, explicitly or implicitly, by the provided sets.
Slices are necessary as they can encode "implicit indexing" -- where a
component is contained in an indexed block. It is natural to return references
to these slices, so they may be accessed and manipulated like any other
component.

Citation
--------
If you use the ``pyomo.dae.flatten`` module in your research, we would appreciate
you citing the following paper, which gives more detail about the motivation for
and examples of using this functinoality.

.. code-block:: bibtex

    @article{parker2023mpc,
    title = {Model predictive control simulations with block-hierarchical differential-algebraic process models},
    journal = {Journal of Process Control},
    volume = {132},
    pages = {103113},
    year = {2023},
    issn = {0959-1524},
    doi = {https://doi.org/10.1016/j.jprocont.2023.103113},
    url = {https://www.sciencedirect.com/science/article/pii/S0959152423002007},
    author = {Robert B. Parker and Bethany L. Nicholson and John D. Siirola and Lorenz T. Biegler},
    }
