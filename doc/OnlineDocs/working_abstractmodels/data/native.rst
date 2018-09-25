Using Standard Data Types
=========================

Defining Constant Values
------------------------

In many cases, Pyomo models can be constructed without :class:`Set <ref
pyomo.core.base.sets.Set>` and :class:`Param <ref
pyomo.core.base.param.Param>` data components.  Native Python data types
class can be simply used to define constant values in Pyomo expressions.
Consequently, Python sets, lists and dictionaries can be used to
construct Pyomo models, as well as a wide range of other Python classes.

.. admonition:: TODO 

    More examples here:  set, list, dict, numpy, pandas.


Initializing Set and Parameter Components
-----------------------------------------

The :class:`Set <ref pyomo.core.base.sets.Set>` and :class:`Param
<ref pyomo.core.base.param.Param>` components used in a Pyomo model
can also be initialized with standard Python data types.  This
enables some modeling efficiencies when manipulating sets (e.g.
when re-using sets for indices), and it supports validation of set
and parameter data values.  The :class:`Set <ref pyomo.core.base.sets.Set>`
and :class:`Param <ref pyomo.core.base.param.Param>` components are
initialized with Python data using the \code{initialize} option.

Set Components
^^^^^^^^^^^^^^

In general, :class:`Set <ref pyomo.core.base.sets.Set>` components
can be initialized with iterable data.  For example, simple sets
can be initialized with:

* list, set and tuple data:

    .. literalinclude:: ../../tests/dataportal/set_initialization_decl2.spy
        :language: python

* generators:

    .. literalinclude:: ../../tests/dataportal/set_initialization_decl3.spy
        :language: python

* numpy arrays:

    .. literalinclude:: ../../tests/dataportal/set_initialization_decl4.spy
        :language: python

Sets can also be indirectly initialized with functions that return
native Python data:

.. literalinclude:: ../../tests/dataportal/set_initialization_decl5.spy
    :language: python

Indexed sets can be initialized with dictionary data where the
dictionary values are iterable data:

.. literalinclude:: ../../tests/dataportal/set_initialization_decl6.spy
    :language: python


Parameter Components
^^^^^^^^^^^^^^^^^^^^

When a parameter is a single value, then a :class:`Param <ref
pyomo.core.base.param.Param>` component can be simply initialized with a
value:

.. literalinclude:: ../../tests/dataportal/param_initialization_decl1.spy
    :language: python

More generally, :class:`Param <ref pyomo.core.base.param.Param>`
components can be initialized with dictionary data where the dictionary
values are single values:

.. literalinclude:: ../../tests/dataportal/param_initialization_decl2.spy
    :language: python

Parameters can also be indirectly initialized with functions that
return native Python data:

.. literalinclude:: ../../tests/dataportal/param_initialization_decl3.spy
    :language: python

