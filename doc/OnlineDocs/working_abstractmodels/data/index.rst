.. _page-managingdata:

Managing Data in AbstractModels
===============================

There are roughly three ways of using data to construct a Pyomo
model:

1. use standard Python objects,

2. initialize a model with data loaded with a
   :class:`~pyomo.environ.DataPortal` object, and

3. load model data from a Pyomo data command file.

Standard Python data objects include native Python data types (e.g.
lists, sets, and dictionaries) as well as standard data formats
like numpy arrays and Pandas data frames.  Standard Python data
objects can be used to define constant values in a Pyomo model, and
they can be used to initialize :class:`~pyomo.environ.Set`
and :class:`~pyomo.environ.Param` components.
However, initializing :class:`~pyomo.environ.Set`
and :class:`~pyomo.environ.Param` components in
this manner provides few advantages over direct use of standard
Python data objects.  (An import exception is that components indexed
by :class:`~pyomo.environ.Set` objects use less
memory than components indexed by native Python data.)

The :class:`~pyomo.environ.DataPortal`
class provides a generic facility for loading data from disparate
sources.  A :class:`~pyomo.environ.DataPortal`
object can load data in a consistent manner, and this data can be
used to simply initialize all :class:`~pyomo.environ.Set`
and :class:`~pyomo.environ.Param` components in
a model.  :class:`~pyomo.environ.DataPortal`
objects can be used to initialize both concrete and abstract models
in a uniform manner, which is important in some scripting applications.
But in practice, this capability is only necessary for abstract
models, whose data components are initialized after being constructed.  (In fact,
all abstract data components in an abstract model are loaded from 
:class:`~pyomo.environ.DataPortal` objects.)

Finally, Pyomo data command files provide a convenient mechanism
for initializing :class:`~pyomo.environ.Set` and
:class:`~pyomo.environ.Param` components with a
high-level data specification.  Data command files can be used with
both concrete and abstract models, though in a different manner.
Data command files are parsed using a :class:`~pyomo.environ.DataPortal` object, which must be done
explicitly for a concrete model.  However, abstract models can load
data from a data command file directly, after the model is constructed.
Again, this capability is only necessary for abstract models, whose
data components are initialized after being constructed.

The following sections provide more detail about how data can be
used to initialize Pyomo models.


.. toctree::
   :maxdepth: 1

   native.rst
   raw_dicts.rst
   datfiles.rst
   dataportals.rst
   storing_data.rst

