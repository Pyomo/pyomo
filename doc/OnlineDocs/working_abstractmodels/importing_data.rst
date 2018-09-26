Data Input for Abstract Models
==============================

This section provides a brief introduction to data commands and options
that are used primarily in abstract models, although they also can be
used on concrete models. For more informatation, see the documentation
with the title Managing Data in Pyomo Models.


Using a Python Dictionary
-------------------------

Data can also be passed to the model `create_instance` method through a
series of nested native Python dictionaries.  The structure begins with
a dictionary of *namespaces*, with the only required entry for the
``None`` namespace.  Each namespace contains a dictionary that maps
component names to dictionaries of component values.  For scalar
components, the required data dictionary maps the implicit index
``None`` to the desired value:

 .. doctest::

    >>> from pyomo.environ import *
    >>> m = AbstractModel()
    >>> m.I = Set()
    >>> m.p = Param()
    >>> m.q = Param(m.I)
    >>> m.r = Param(m.I, m.I, default=0)
    >>> data = {None: {
    ...     'I': {None: [1,2,3]},
    ...     'p': {None: 100},
    ...     'q': {1: 10, 2:20, 3:30},
    ...     'r': {(1,1): 110, (1,2): 120, (2,3): 230},
    ... }}
    >>> i = m.create_instance(data)
    >>> i.pprint()
    2 Set Declarations
        I : Dim=0, Dimen=1, Size=3, Domain=None, Ordered=False, Bounds=(1, 3)
            [1, 2, 3]
        r_index : Dim=0, Dimen=2, Size=9, Domain=None, Ordered=False, Bounds=None
            Virtual
    <BLANKLINE>
    3 Param Declarations
        p : Size=1, Index=None, Domain=Any, Default=None, Mutable=False
            Key  : Value
            None :   100
        q : Size=3, Index=I, Domain=Any, Default=None, Mutable=False
            Key : Value
              1 :    10
              2 :    20
              3 :    30
        r : Size=9, Index=r_index, Domain=Any, Default=0, Mutable=False
            Key    : Value
            (1, 1) :   110
            (1, 2) :   120
            (2, 3) :   230
    <BLANKLINE>
    5 Declarations: I p q r_index r


Data Command Files
------------------

The following commands can be used in data command files:

* `set` declares set data,
* `param` declares a table of parameter data, which can also include the
  declaration of the set data used to index parameter data,
* `load` loads set and parameter data from an external data source such
  as ASCII table files, CSV files, ranges in spreadsheets, and database
  tables,
* `table` loads set and parameter data from a table,
* `include` specifies a data command file that is to be processed immediately,
* the `data` and `end` commands do not perform any actions, but they
  provide compatibility with AMPL scripts that define data commands, and
* `namespace` defines groupings of data commands.

The syntax of the `set` and `param` data commands are adapted from
AMPL's data commands.  However, other Pyomo data commands do not
directly correspond to AMPL data commands. In particular, Pyomo's
`table` command was introduced to work around semantic ambiguities in
the `param` command.  Pyomo's `table` command does not correspond to
AMPL's `table` command.  Instead, the `load` command mimics AMPL's
`table` command with a simplified syntax.

.. warning::

   The data command file was initially developed to provide
   compatability in data formats between Pyomo and AMPL.  However, these
   data formats continue to diverge in their syntax and semantics.
   Simple examples using `set` and `param` data commands are likely to
   work for both AMPL and Pyomo, particularly with abstract Pyomo
   models.  But in general a user should expect to need to adapt their
   AMPL data command files for use with Pyomo.

DataPortal Objects
------------------

The `load` and `store` Pyomo data commands can be used to load set and
table data from a variety of data sources.  Pyomo's `DataPortal` object
provides this same functionality for users who work with Python scripts.
A `DataPortal` object manages the process of loading data from different
data sources, and it is used to construct model instances in a standard
manner.  Similarly, a `DataPortal` object can be used to store model
data externally in a standard manner.

Loading Data
************

The `load` method can be used to load data into Pyomo models from a
variety of sources and formats.  The most common format is a table
representation of set and parameter data.  For more informatation, see
the documentation with the title Managing Data in Pyomo Models.


