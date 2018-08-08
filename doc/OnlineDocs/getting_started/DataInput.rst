Data Input for Abstract Models
==============================

This section provides a brief introduction to data commands and options
that are used primarily in abstract models, although they
also can be used on concrete models. For more informatation,
see the documentation with the title Managing Data in Pyomo Models.


Data Command Files
------------------

The following commands can be used in data command files:

* `set` declares set data,
* `param` declares a table of parameter data, which can also include the declaration of the set data used to index parameter data,
* `load` loads set and parameter data from an external data source such as ASCII table files, CSV files, ranges in spreadsheets, and database tables,
* `table` loads set and parameter data from a table,
* `include` specifies a data command file that is to be processed immediately,
* the `data` and `end` commands do not perform any actions, but they provide compatibility with AMPL scripts that define data commands, and
* `namespace` defines groupings of data commands.

The syntax of the `set` and `param` data commands are adapted from
AMPL's data commands.  However, other Pyomo data commands do not
directly correspond to AMPL data commands. In particular, Pyomo's
`table` command was introduced to work around semantic ambiguities
in the `param` command.  Pyomo's `table` command does not correspond
to AMPL's `table` command.  Instead, the `load` command mimics
AMPL's `table` command with a simplified syntax.

WARNING: The data command file was initially developed to provide
compatability in data formats between Pyomo and AMPL.  However,
these data formats continue to diverge in their syntax and semantics.
Simple examples using `set` and `param` data commands are likely
to work for both AMPL and Pyomo, particularly with abstract Pyomo
models.  But in general a user should expect to need to adapt their
AMPL data command files for use with Pyomo.

DataPortal Objects
------------------

The `load` and `store` Pyomo data commands can be used to load
set and table data from a variety of data sources.  Pyomo's
`DataPortal` object provides this same functionality for users who
work with Python scripts.  A `DataPortal` object manages the process
of loading data from different data sources, and it is used to
construct model instances in a standard manner.  Similarly, a
`DataPortal` object can be used to store model data externally in
a standard manner.


Loading Data
************

The `load` method can be used to load data into Pyomo models from a variety of 
sources and formats.  The most common format is a table representation of set and
parameter data.   For more informatation,
see the documentation with the title Managing Data in Pyomo Models.


