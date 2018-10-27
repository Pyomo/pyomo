.. _rappersection:

rapper: a PySP wrapper
======================

This is an advanced topic.

The `pyomo.pysp.util.rapper` package is built on the Pyomo optimization
modeling language ([PyomoJournal]_, [PyomoBookII]_) to provide a thin
wrapper for some functionality of PySP [PySPJournal]_ associated with
the `runef` and `runph` commands. The package is designed mainly for
experienced Python programmers who are users of a Pyomo `ConcreteModel` in PySP
and who want to embed the solution process is simple scripts. There
is also support for users of a Pyomo `AbstractModel`. Note that
callback functions are also supported for some aspects of PySP, which
is somewhat orthogonal to the functionality provided by
`pyomo.pysp.util.rapper`.

.. toctree::
   :maxdepth: 2

   demorapper.rst
   stochsolverapi.rst
   Abstractrapper.rst
   rap.rst
   bibliography.rst

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
