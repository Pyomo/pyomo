Data Command Files
==================

.. note::

    The discussion and presentation below are adapted from Chapter 6 of
    the "Pyomo Book" [PyomoBookII]_.  The discussion of the
    :class:`DataPortal <ref pyomo.core.base.DataPortal.DataPortal>`
    class uses these same examples to illustrate how data can be loaded
    into Pyomo models within Python scripts (see the
    :ref:`page-dataportals` section).

Model Data
----------

Pyomo's *data command files* employ a domain-specific language whose
syntax closely resembles the syntax of AMPL's data commands [AMPL]_.  A
data command file consists of a sequence of commands that either (a)
specify set and parameter data for a model, or (b) specify where such
data is to be obtained from external sources (e.g. table files, CSV
files, spreadsheets and databases).

The following commands are used to declare data:

* The ``set`` command declares set data.

* The ``param`` command declares a table of parameter data, which
  can also include the declaration of the set data used to index the
  parameter data.

* The ``table`` command declares a two-dimensional table of parameter
  data.

* The ``load`` command defines how set and parameter data is loaded from
  external data sources, including ASCII table files, CSV files, XML
  files, YAML files, JSON files, ranges in spreadsheets, and database
  tables.

The following commands are also used in data command files:

* The ``include`` command specifies a data command file that is
  processed immediately.

* The ``data`` and ``end`` commands do not perform any actions, but they
  provide compatibility with AMPL scripts that define data commands.

* The ``namespace`` keyword allows data commands to be organized into
  named groups that can be enabled or disabled during model
  construction.

The following data types can be represented in a data command file:

* **Numeric value**: Any Python numeric value (e.g. integer, float,
  scientific notation, or boolean).

* **Simple string**: A sequence of alpha-numeric characters.

* **Quoted string**: A simple string that is included in a pair of
  single or double quotes.  A quoted string can include quotes within
  the quoted string.

Numeric values are automatically converted to Python integer or floating
point values when a data command file is parsed. Additionally, if a
quoted string can be intepreted as a numeric value, then it will be
converted to Python numeric types when the data is parsed.  For example,
the string "100" is converted to a numeric value automatically.

.. warning::

    Pyomo data commands do *not* exactly correspond to AMPL data
    commands.  The ``set`` and ``param`` commands are designed to
    closely match AMPL's syntax and semantics, though these commands
    only support a subset of the corresponding declarations in AMPL.
    However, other Pyomo data commands are not generally designed to
    match the semantics of AMPL.

.. note::

    Pyomo data commands are terminated with a semicolon, and the syntax
    of data commands does not depend on whitespace.  Thus, data commands
    can be broken across multiple lines -- newlines and tab characters
    are ignored -- and data commands can be formatted with whitespace
    with few restrictions.


The ``set`` Command
-------------------

Simple Sets
^^^^^^^^^^^

The ``set`` data command explicitly specifies the members of either a
single set or an array of sets, i.e., an indexed set.  A single set is
specified with a list of data values that are included in this set.  The
formal syntax for the set data command is:

::

    set <setname> := [<value>] ... ;

A set may be empty, and it may contain any combination of numeric and
non-numeric string values.  For example, the following are valid ``set``
commands:

.. literalinclude:: ../../tests/data/set1.dat
    :language: python


Sets of Tuple Data
^^^^^^^^^^^^^^^^^^

The ``set`` data command can also specify tuple data with the standard
notation for tuples.  For example, suppose that set ``A`` contains
3-tuples:

.. literalinclude:: ../../tests/data/set2_decl.spy
    :language: python

The following ``set`` data command then specifies that ``A`` is the set
containing the tuples ``(1,2,3)`` and ``(4,5,6)``:

.. literalinclude:: ../../tests/data/set2a.dat
    :language: none

Alternatively, set data can simply be listed in the order that the tuple
is represented:

.. literalinclude:: ../../tests/data/set2.dat
    :language: none

Obviously, the number of data elements specified using this syntax
should be a multiple of the set dimension.

Sets with 2-tuple data can also be specified in a matrix denoting set
membership.  For example, the following ``set`` data command declares
2-tuples in ``A`` using plus (``+``) to denote valid tuples and minus
(``-``) to denote invalid tuples:

.. literalinclude:: ../../tests/data/set4.dat
    :language: none

This data command declares the following five 2-tuples: ``('A1',1)``,
``('A1',2)``, ``('A2',3)``, ``('A3',2)``, and ``('A4',1)``.

Finally, a set of tuple data can be concisely represented with tuple
*templates* that represent a *slice* of tuple data.  For example,
suppose that the set ``A`` contains 4-tuples:

.. literalinclude:: ../../tests/data/set5_decl.spy
    :language: python

The following ``set`` data command declares groups of tuples that are
defined by a template and data to complete this template:

.. literalinclude:: ../../tests/data/set5.dat
    :language: none

A tuple template consists of a tuple that contains one or more asterisk
(``*``) symbols instead of a value.  These represent indices where the
tuple value is replaced by the values from the list of values that
follows the tuple template.  In this example, the following tuples are
in set ``A``:

.. literalinclude:: ../../tests/data/set5.txt
    :language: none

Set Arrays
^^^^^^^^^^

The ``set`` data command can also be used to declare data for a set
array.  Each set in a set array must be declared with a separate ``set``
data command with the following syntax:

::

    set <set-name>[<index>] := [<value>] ... ;

Because set arrays can be indexed by an arbitrary set, the index value
may be a numeric value, a non-numeric string value, or a comma-separated
list of string values.

Suppose that a set ``A`` is used to index a set ``B`` as follows:

.. literalinclude:: ../../tests/data/set3_decl.spy
    :language: python

Then set ``B`` is indexed using the values declared for set ``A``:

.. literalinclude:: ../../tests/data/set3.dat
    :language: none

The ``param`` Command
---------------------

Simple or non-indexed parameters are declared in an obvious way, as
shown by these examples:

.. literalinclude:: ../../tests/data/param1.dat
    :language: none

Parameters can be defined with numeric data, simple strings and quoted
strings.  Note that parameters cannot be defined without data, so there
is no analog to the specification of an empty set.

One-dimensional Parameter Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most parameter data is indexed over one or more sets, and there are a
number of ways the ``param`` data command can be used to specify indexed
parameter data.  One-dimensional parameter data is indexed over a single
set.  Suppose that the parameter ``B`` is a parameter indexed by the set
``A``:

.. literalinclude:: ../../tests/data/param2_decl.spy
    :language: python

A ``param`` data command can specify values for ``B`` with a list of
index-value pairs:

.. literalinclude:: ../../tests/data/param2.dat
    :language: none

Because whitespace is ignored, this example data command file can be
reorganized to specify the same data in a tabular format:

.. literalinclude:: ../../tests/data/param2a.dat
    :language: none

Multiple parameters can be defined using a single ``param`` data
command. For example, suppose that parameters ``B``, ``C``, and ``D``
are one-dimensional parameters all indexed by the set ``A``:

.. literalinclude:: ../../tests/data/param3_decl.spy
    :language: python

Values for these parameters can be specified using a single ``param``
data command that declares these parameter names followed by a list of
index and parameter values:

.. literalinclude:: ../../tests/data/param3.dat
    :language: none

The values in the ``param`` data command are interpreted as a list of
sublists, where each sublist consists of an index followed by the
corresponding numeric value.

Note that parameter values do not need to be defined for all indices.
For example, the following data command file is valid:

.. literalinclude:: ../../tests/data/param3a.dat
    :language: none

The index ``g`` is omitted from the ``param`` command, and consequently
this index is not valid for the model instance that uses this data.
More complex patterns of missing data can be specified using the period
(``.``) symbol to indicate a missing value.  This syntax is useful when
specifying multiple parameters that do not necessarily have the same
index values:

.. literalinclude:: ../../tests/data/param3b.dat
    :language: none

This example provides a concise representation of parameters that share
a common index set while using different index values.

Note that this data file specifies the data for set ``A`` twice:
(1) when ``A`` is defined and (2) implicitly when the parameters are
defined.  An alternate syntax for ``param`` allows the user to concisely
specify the definition of an index set along with associated parameters:

.. literalinclude:: ../../tests/data/param3c.dat
    :language: none

Finally, we note that default values for missing data can also be
specified using the ``default`` keyword:

.. literalinclude:: ../../tests/data/param4.dat
    :language: none

Note that default values can only be specified in ``param`` commands
that define values for a single parameter.


Multi-Dimensional Parameter Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multi-dimensional parameter data is indexed over either multiple sets or
a single multi-dimensional set.  Suppose that parameter ``B`` is a
parameter indexed by set ``A`` that has dimension 2:

.. literalinclude:: ../../tests/data/param5_decl.spy
    :language: python

The syntax of the ``param`` data command remains essentially the same
when specifying values for ``B`` with a list of index and parameter
values:

.. literalinclude:: ../../tests/data/param5.dat
    :language: none

Missing and default values are also handled in the same way with
multi-dimensional index sets:

.. literalinclude:: ../../tests/data/param5a.dat
    :language: none

Similarly, multiple parameters can defined with a single ``param`` data
command.  Suppose that parameters ``B``, ``C``, and ``D`` are parameters
indexed over set ``A`` that has dimension 2:

.. literalinclude:: ../../tests/data/param6_decl.spy
    :language: python

These parameters can be defined with a single ``param`` command that
declares the parameter names followed by a list of index and parameter
values:

.. literalinclude:: ../../tests/data/param6.dat
    :language: none

Similarly, the following ``param`` data command defines the index set
along with the parameters:

.. literalinclude:: ../../tests/data/param6a.dat
    :language: none

The ``param`` command also supports a matrix syntax for specifying the
values in a parameter that has a 2-dimensional index.  Suppose parameter
``B`` is indexed over set ``A`` that has dimension 2:

.. literalinclude:: ../../tests/data/param7a_decl.spy
    :language: python

The following ``param`` command defines a matrix of parameter values:

.. literalinclude:: ../../tests/data/param7a.dat
    :language: none

Additionally, the following syntax can be used to specify a transposed
matrix of parameter values:

.. literalinclude:: ../../tests/data/param7b.dat
    :language: none

This functionality facilitates the presentation of parameter data in a
natural format.  In particular, the transpose syntax may allow the
specification of tables for which the rows comfortably fit within a
single line.  However, a matrix may be divided column-wise into shorter
rows since the line breaks are not significant in Pyomo data commands.

For parameters with three or more indices, the parameter data values may
be specified as a series of slices.  Each slice is defined by a template
followed by a list of index and parameter values.  Suppose that
parameter ``B`` is indexed over set ``A`` that has dimension 4:

.. literalinclude:: ../../tests/data/param8a_decl.spy
    :language: python

The following ``param`` command defines a matrix of parameter values
with multiple templates:

.. literalinclude:: ../../tests/data/param8a.dat
    :language: none

The ``B`` parameter consists of four values: ``B[a,1,a,1]=10``,
``B[b,1,b,1]=20``, ``B[a,2,a,2]=30``, and ``B[b,2,b,2]=40``.

The ``table`` Command
---------------------

The ``table`` data command explicitly specifies a two-dimensional array
of parameter data.  This command provides a more flexible and complete
data declaration than is possible with a ``param`` declaration.  The
following example illustrates a simple ``table`` command that declares
data for a single parameter:

.. literalinclude:: ../../tests/data/table0.dat
    :language: none

The parameter ``M`` is indexed by column ``A``, which must be
pre-defined unless declared separately (see below).  The column labels
are provided after the colon and before the colon-equal (``:=``).
Subsequently, the table data is provided.  The syntax is not sensitive
to whitespace, so the following is an equivalent ``table`` command:

.. literalinclude:: ../../tests/data/table1.dat
    :language: none

Multiple parameters can be declared by simply including additional
parameter names.  For example:

.. literalinclude:: ../../tests/data/table2.dat
    :language: none

This example declares data for the ``M`` and ``N`` parameters, which
have different indexing columns.  The indexing columns represent set
data, which is specified separately.  For example:

.. literalinclude:: ../../tests/data/table3.dat
    :language: none

This example declares data for the ``M`` and ``N`` parameters, along
with the ``A`` and ``Z`` indexing sets.  The correspondence between the
index set ``Z`` and the indices of parameter ``N`` can be made more
explicit by indexing ``N`` by ``Z``:

.. literalinclude:: ../../tests/data/table4.dat
    :language: none

Set data can also be specified independent of parameter data:

.. literalinclude:: ../../tests/data/table5.dat
    :language: none

.. warning::

    If a ``table`` command does not explicitly indicate the indexing
    sets, then these are assumed to be initialized separately.  A
    ``table`` command can separately initialize sets and parameters in a
    Pyomo model, and there is no presumed association between the data
    that is initialized.  For example, the ``table`` command initializes
    a set ``Z`` and a parameter ``M`` that are not related:

    .. literalinclude:: ../../tests/data/table7.dat
        :language: none

Finally, simple parameter values can also be specified with a ``table``
command:

.. literalinclude:: ../../tests/data/table6.dat
    :language: none

The previous examples considered examples of the ``table`` command where
column labels are provided.  The ``table`` command can also be used
without column labels.  For example, the first example can be revised to
omit column labels as follows:

.. literalinclude:: ../../tests/data/table0.ul.dat
    :language: none

The ``columns=4`` is a keyword-value pair that defines the number of
columns in this table; this must be explicitly specified in tables
without column labels.  The default column labels are integers starting
from ``1``; the labels are columns ``1``, ``2``, ``3``, and ``4`` in
this example.  The ``M`` parameter is indexed by column ``1``.  The
braces syntax declares the column where the ``M`` data is provided.

Similarly, set data can be declared referencing the integer column
labels:

.. literalinclude:: ../../tests/data/table3.ul.dat
    :language: none

Declared set names can also be used to index parameters:

.. literalinclude:: ../../tests/data/table4.ul.dat
    :language: none

Finally, we compare and contrast the ``table`` and ``param`` commands.
Both commands can be used to declare parameter and set data, and both
commands can be used to declare a simple parameter.  However, there are
some important differences between these data commands:

* The ``param`` command can declare a single set that is used to index
  one or more parameters.  The ``table`` command can declare data for
  any number of sets, independent of whether they are used to index
  parameter data.

* The ``param`` command can declare data for multiple parameters only if
  they share the same index set.  The ``table`` command can declare data
  for any number of parameters that are may be indexed separately.

* The ``table`` syntax unambiguously describes the dimensionality of
  indexing sets.  The ``param`` command must be interpreted with a model
  that provides the dimension of the indexing set.

This last point provides a key motivation for the ``table`` command.
Specifically, the ``table`` command can be used to reliably initialize
concrete models using Pyomo's :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` class.  By contrast, the
``param`` command can only be used to initialize concrete models with
parameters that are indexed by a single column (i.e., a simple set).

The ``load`` Command
--------------------

The ``load`` command provides a mechanism for loading data from a
variety of external tabular data sources.  This command loads a table of
data that represents set and parameter data in a Pyomo model.  The table
consists of rows and columns for which all rows have the same length,
all columns have the same length, and the first row represents labels
for the column data.

The ``load`` command can load data from a variety of different external
data sources:

* **TAB File**: A text file format that uses whitespace to separate
  columns of values in each row of a table.

* **CSV File**: A text file format that uses comma or other delimiters
  to separate columns of values in each row of a table.

* **XML File**: An extensible markup language for documents and data
  structures.  XML files can represent tabular data.

* **Excel File**: A spreadsheet data format that is primarily used by
  the Microsoft Excel application.

* **Database**: A relational database.

This command uses a *data manager* that coordinates how data is
extracted from a specified *data source*.  In this way, the ``load``
command provides a generic mechanism that enables Pyomo models to
interact with standard data repositories that are maintained in an
application-specific manner.

Simple Load Examples
^^^^^^^^^^^^^^^^^^^^

The simplest illustration of the ``load`` command is specifying data for
an indexed parameter.  Consider the file ``Y.tab``:

.. literalinclude:: ../../tests/data/Y.tab
    :language: none

This file specifies the values of parameter ``Y`` which is indexed by
set ``A``.  The following ``load`` command loads the parameter data:

.. literalinclude:: ../../tests/data/import1.tab.dat
    :language: none

The first argument is the filename.  The options after the colon
indicate how the table data is mapped to model data.  Option ``[A]``
indicates that set ``A`` is used as the index, and option ``Y``
indicates the parameter that is initialized.

Similarly, the following load command loads both the parameter data as
well as the index set ``A``:

.. literalinclude:: ../../tests/data/import2.tab.dat
    :language: none

The difference is the specification of the index set, ``A=[A]``, which
indicates that set ``A`` is initialized with the index loaded from the
ASCII table file.

Set data can also be loaded from a ASCII table file that contains a
single column of data:

.. literalinclude:: ../../tests/data/A.tab
    :language: none

The ``format`` option must be specified to denote the fact that the
relational data is being interpreted as a set:

.. literalinclude:: ../../tests/data/import3.tab.dat
    :language: none

Note that this allows for specifying set data that contains tuples.
Consider file ``C.tab``:

.. literalinclude:: ../../tests/data/C.tab
    :language: none

A similar ``load`` syntax will load this data into set ``C``:

.. literalinclude:: ../../tests/data/import4.tab.dat
    :language: none

Note that this example requires that ``C`` be declared with dimension
two.

Load Syntax Options
^^^^^^^^^^^^^^^^^^^

The syntax of the ``load`` command is broken into two parts.  The first
part ends with the colon, and it begins with a filename, database URL,
or DSN (data source name).  Additionally, this first part can contain
option value pairs.  The following options are recognized:

.. list-table::

    * - ``format``
      - A string that denotes how the relational table is interpreted
    * - ``password``
      - The password that is used to access a database
    * - ``query``
      - The query that is used to request data from a database
    * - ``range``
      - The subset of a spreadsheet that is requested\index{spreadsheet}
    * - ``user``
      - The user name that is used to access the data source
    * - ``using``
      - The data manager that is used to process the data source
    * - ``table``
      - The database table that is requested

The ``format`` option is the only option that is required for all data
managers.  This option specifies how a relational table is interpreted
to represent set and parameter data.  If the ``using`` option is
omitted, then the filename suffix is used to select the data manager.
The remaining options are specific to spreadsheets and relational
databases (see below).

The second part of the ``load`` command consists of the specification of
column names for indices and data.  The remainder of this section
describes different specifications and how they define how data is
loaded into a model.  Suppose file ``ABCD.tab`` defines the following
relational table:

.. literalinclude:: ../../tests/data/ABCD.tab
    :language: none

There are many ways to interpret this relational table.  It could
specify a set of 4-tuples, a parameter indexed by 3-tuples, two
parameters indexed by 2-tuples, and so on.  Additionally, we may wish to
select a subset of this table to initialize data in a model.
Consequently, the ``load`` command provides a variety of syntax options
for specifying how a table is interpreted.

A simple specification is to interpret the relational table as a set:

.. literalinclude:: ../../tests/data/ABCD1.dat
    :language: none

Note that ``Z`` is a set in the model that the data is being loaded
into.  If this set does not exist, an error will occur while loading
data from this table.

Another simple specification is to interpret the relational table as a
parameter with indexed by 3-tuples:

.. literalinclude:: ../../tests/data/ABCD2.dat
    :language: none

Again, this requires that ``D`` be a parameter in the model that the
data is being loaded into.  Additionally, the index set for ``D`` must
contain the indices that are specified in the table.  The ``load``
command also allows for the specification of the index set:

.. literalinclude:: ../../tests/data/ABCD3.dat
    :language: none

This specifies that the index set is loaded into the ``Z`` set in the
model.  Similarly, data can be loaded into another parameter than what
is specified in the relational table:

.. literalinclude:: ../../tests/data/ABCD4.dat
    :language: none

This specifies that the index set is loaded into the ``Z`` set and that
the data in the ``D`` column in the table is loaded into the ``Y``
parameter.

This syntax allows the ``load`` command to provide an arbitrary
specification of data mappings from columns in a relational table into
index sets and parameters.  For example, suppose that a model is defined
with set ``Z`` and parameters ``Y`` and ``W``:

.. literalinclude:: ../../tests/data/ABCD5_decl.spy
    :language: python

Then the following command defines how these data items are loaded using
columns ``B``, ``C`` and ``D``:

.. literalinclude:: ../../tests/data/ABCD5.dat
    :language: none

When the ``using`` option is omitted the data manager is inferred from
the filename suffix.  However, the filename suffix does not always
reflect the format of the data it contains.  For example, consider the
relational table in the file ``ABCD.txt``:

.. literalinclude:: ../../tests/data/ABCD.txt
    :language: none

We can specify the ``using`` option to load from this file into
parameter ``D`` and set ``Z``:

.. literalinclude:: ../../tests/data/ABCD6.dat
    :language: none

.. note::

    The data managers supported by Pyomo can be listed with the
    ``pyomo help`` subcommand
    
    ::

        pyomo help --data-managers

    The following data managers are supported in Pyomo 5.1:
    
        .. literalinclude:: ../../tests/data/data_managers.txt
            :language: none

Interpreting Tabular Data
^^^^^^^^^^^^^^^^^^^^^^^^^

By default, a table is interpreted as columns of one or more parameters
with associated index columns.  The ``format`` option can be used to
specify other interpretations of a table:

.. list-table::

    * - ``array``
      - The table is a matrix representation of a two dimensional
        parameter.
    * - ``param``
      - The data is a simple parameter value.
    * - ``set``
      - Each row is a set element.
    * - ``set_array`` 
      - The table is a matrix representation of a set of 2-tuples.
    * - ``transposed_array`` 
      - The table is a transposed matrix representation of a two
        dimensional parameter.

We have previously illustrated the use of the ``set`` format value to
interpret a relational table as a set of values or tuples.  The
following examples illustrate the other format values.

A table with a single value can be interpreted as a simple parameter
using the ``param`` format value.  Suppose that ``Z.tab`` contains the
following table:

.. literalinclude:: ../../tests/data/Z.tab
    :language: none

The following load command then loads this value into parameter ``p``:

.. literalinclude:: ../../tests/data/import6.tab.dat
    :language: none

Sets with 2-tuple data can be represented with a matrix format that
denotes set membership.  The ``set_array`` format value interprets a
relational table as a matrix that defines a set of 2-tuples where ``+``
denotes a valid tuple and ``-`` denotes an invalid tuple.  Suppose that
``D.tab`` contains the following relational table:

.. literalinclude:: ../../tests/data/D.tab
    :language: none

Then the following load command loads data into set ``B``:

.. literalinclude:: ../../tests/data/import5.tab.dat
    :language: none

This command declares the following 2-tuples: ``('A1',1)``,
``('A2',2)``, and ``('A3',3)``.

Parameters with 2-tuple indices can be interpreted with a matrix format
that where rows and columns are different indices.  Suppose that
``U.tab`` contains the following table:

.. literalinclude:: ../../tests/data/U.tab
    :language: none

Then the following load command loads this value into parameter ``U``
with a 2-dimensional index using the ``array`` format value.:

.. literalinclude:: ../../tests/data/import7.tab.dat
    :language: none

The ``transpose_array`` format value also interprets the table as a
matrix, but it loads the data in a transposed format:

.. literalinclude:: ../../tests/data/import8.tab.dat
    :language: none

Note that these format values do not support the initialization of the
index data.

Loading from Spreadsheets and Relational Databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many of the options for the ``load`` command are specific to
spreadsheets and relational databases.  The ``range`` option is used to
specify the range of cells that are loaded from a spreadsheet.  The
range of cells represents a table in which the first row of cells
defines the column names for the table.

Suppose that file ``ABCD.xls`` contains the range ``ABCD`` that is shown
in the following figure:

.. image:: ABCD.png

The following command loads this data to initialize parameter ``D`` and
index ``Z``:

.. literalinclude:: ../../tests/data/ABCD7.dat
    :language: none

Thus, the syntax for loading data from spreadsheets only differs from
CSV and ASCII text files by the use of the ``range`` option.

When loading from a relational database, the data source specification
is a filename or data connection string.  Access to a database may be
restricted, and thus the specification of ``username`` and ``password``
options may be required.  Alternatively, these options can be specified
within a data connection string.

A variety of database interface packages are available within Python.
The ``using`` option is used to specify the database interface package
that will be used to access a database.  For example, the ``pyodbc``
interface can be used to connect to Excel spreadsheets.  The following
command loads data from the Excel spreadsheet ``ABCD.xls`` using the
``pyodbc`` interface.  The command loads this data to initialize
parameter ``D`` and index ``Z``:

.. literalinclude:: ../../tests/data/ABCD8.dat
    :language: none

The ``using`` option specifies that the ``pyodbc`` package will be
used to connect with the Excel spreadsheet.  The ``table`` option
specifies that the table ``ABCD`` is loaded from this spreadsheet.
Similarly, the following command specifies a data connection string
to specify the ODBC driver explicitly:

.. literalinclude:: ../../tests/data/ABCD9.dat
    :language: none

ODBC drivers are generally tailored to the type of data source that
they work with;  this syntax illustrates how the ``load`` command
can be tailored to the details of the database that a user is working
with.

The previous examples specified the ``table`` option, which declares the
name of a relational table in a database.  Many databases support the
Structured Query Language (SQL), which can be used to dynamically
compose a relational table from other tables in a database.  The classic
diet problem will be used to illustrate the use of SQL queries to
initialize a Pyomo model.  In this problem, a customer is faced with the
task of minimizing the cost for a meal at a fast food restaurant -- they
must purchase a sandwich, side, and a drink for the lowest cost.  The
following is a Pyomo model for this problem:

.. literalinclude:: ../../tests/data/diet1.py
    :language: python

Suppose that the file ``diet1.sqlite`` be a SQLite database file that
contains the following data in the ``Food`` table:

.. list-table::
    :header-rows: 1

    * - **FOOD** 
      - **cost**

    * - Cheeseburger
      - 1.84

    * - Ham Sandwich 
      - 2.19

    * - Hamburger 
      - 1.84

    * - Fish Sandwich 
      - 1.44

    * - Chicken Sandwich 
      - 2.29

    * - Fries 
      - 0.77

    * - Sausage Biscuit 
      - 1.29

    * - Lowfat Milk 
      - 0.60

    * - Orange Juice 
      - 0.72

In addition, the ``Food`` table has two additional columns, ``f_min``
and ``f_max``, with no data for any row. These columns exist to match
the structure for the parameters used in the model.

We can solve the ``diet1`` model using the Python definition in
``diet1.py`` and the data from this database.  The file
``diet.sqlite.dat`` specifies a ``load`` command that uses that
``sqlite3`` data manager and embeds a SQL query to retrieve the data:

.. literalinclude:: ../../tests/data/diet.sqlite.dat
    :language: none

The PyODBC driver module will pass the SQL query through an Access ODBC
connector, extract the data from the ``diet1.mdb`` file, and return it
to Pyomo. The Pyomo ODBC handler can then convert the data received into
the proper format for solving the model internally.  More complex SQL
queries are possible, depending on the underlying database and ODBC
driver in use. However, the name and ordering of the columns queried are
specified in the Pyomo data file; using SQL wildcards (e.g., ``SELECT
*``) or column aliasing (e.g., ``SELECT f AS FOOD``) may cause errors in
Pyomo's mapping of relational data to parameters.

The ``include`` Command
-----------------------

The ``include`` command allows a data command file to execute data
commands from another file.  For example, the following command file
executes data commands from ``ex1.dat`` and then ``ex2.dat``:

.. literalinclude:: ../../tests/data/ex.dat
    :language: none

Pyomo is sensitive to the order of execution of data commands, since
data commands can redefine set and parameter values.  The ``include``
command respects this data ordering; all data commands in the included
file are executed before the remaining data commands in the current file
are executed.

The ``namespace`` Keyword
-------------------------

The ``namespace`` keyword is not a data command, but instead it is used
to structure the specification of Pyomo's data commands.  Specifically,
a namespace declaration is used to group data commands and to provide a
group label. Consider the following data command file:

.. literalinclude:: ../../tests/data/namespace1.dat
    :language: none

This data file defines two namespaces: ``ns1`` and ``ns2`` that
initialize a set ``C``.  By default, data commands contained within a
namespace are ignored during model construction; when no namespaces are
specified, the set ``C`` has values ``1,2,3``.  When namespace ``ns1``
is specified, then the set ``C`` values are overridden with the set
``4,5,6``.

