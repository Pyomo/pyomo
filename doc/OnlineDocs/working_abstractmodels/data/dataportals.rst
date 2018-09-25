.. _page-dataportals:

Data Portals
============

Pyomo's :class:`DataPortal <ref pyomo.core.base.DataPortal.DataPortal>`
class standardizes the process of constructing model instances by
managing the process of loading data from different data sources in a
uniform manner.  A :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object can load data from the
following data sources:

* **TAB File**: A text file format that uses whitespace to separate
  columns of values in each row of a table.

* **CSV File**: A text file format that uses comma or other delimiters
  to separate columns of values in each row of a table.

* **JSON File**: A popular lightweight data-interchange format that is
  easily parsed.

* **YAML File**:  A human friendly data serialization standard.

* **XML File**: An extensible markup language for documents and data
  structures.  XML files can represent tabular data.

* **Excel File**: A spreadsheet data format that is primarily used by
  the Microsoft Excel application.

* **Database**: A relational database.

* **DAT File**: A Pyomo data command file.

Note that most of these data formats can express tabular data.

.. warning::

    The :class:`DataPortal <ref pyomo.core.base.DataPortal.DataPortal>`
    class requires the installation of Python packages to support some
    of these data formats:

    * **YAML File**: ``pyyaml``

    * **Excel File**: ``win32com``, ``openpyxl`` or ``xlrd``

        These packages support different data Excel data formats:
        the ``win32com`` package supports ``.xls``, ``.xlsm``
        and ``.xlsx``, the ``openpyxl`` package supports
        ``.xlsx`` and the ``xlrd`` package supports ``.xls``.


    * **Database**: ``pyodbc``, ``pypyodbc``, ``sqlite3`` or ``pymysql``

        These packages support different database interface APIs:
        the ``pyodbc`` and ``pypyodbc`` packages support the ODBC
        database API, the ``sqlite3`` package uses the SQLite C
        library to directly interface with databases using the
        DB-API 2.0 specification, and ``pymysql`` is a pure-Python
        MySQL client.

:class:`DataPortal <ref pyomo.core.base.DataPortal.DataPortal>` objects
can be used to initialize both concrete and abstract Pyomo models.
Consider the file ``A.tab``, which defines a simple set with a tabular
format:

.. literalinclude:: ../../tests/dataportal/A.tab
    :language: none

The ``load`` method is used to load data into a :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object.  Components in a
concrete model can be explicitly initialized with data loaded by a
:class:`DataPortal <ref pyomo.core.base.DataPortal.DataPortal>` object:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_concrete1.spy
    :language: python

All data needed to initialize an abstract model *must* be provided by a
:class:`DataPortal <ref pyomo.core.base.DataPortal.DataPortal>` object,
and the use of the :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object to initialize components
is automated for the user:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_load.spy
    :language: python

Note the difference in the execution of the ``load`` method in these two
examples: for concrete models data is loaded by name and the format must
be specified, and for abstract models the data is loaded by component,
from which the data format can often be inferred.

The ``load`` method opens the data file, processes it, and loads the
data in a format that can be used to construct a model instance.  The
``load`` method can be called multiple times to load data for different
sets or parameters, or to override data processed earlier.  The ``load``
method takes a variety of arguments that define how data is loaded:

* ``filename``: This option specifies the source data file.

* ``format``: This option specifies the how to interpret data within a
  table.  Valid formats are: ``set``, ``set_array``, ``param``,
  ``table``, ``array``, and ``transposed_array``.

* ``set``: This option is either a string or model compent that defines
  a set that will be initialized with this data.

* ``param``: This option is either a string or model compent that
  defines a parameter that will be initialized with this data.  A list
  or tuple of strings or model components can be used to define multiple
  parameters that are initialized.

* ``index``: This option is either a string or model compent that
  defines an index set that will be initialized with this data.

* ``using``: This option specifies the Python package used to load this
  data source.  This option is used when loading data from databases.

* ``select``: This option defines the columns that are selected from the
  data source.  The column order may be changed from the data source,
  which allows the :class:`DataPortal <ref
  pyomo.core.base.DataPortal.DataPortal>` object to define

* ``namespace``: This option defines the data namespace that will
  contain this data.

The use of these options is illustrated below.

The :class:`DataPortal <ref pyomo.core.base.DataPortal.DataPortal>`
class also provides a simple API for accessing set and parameter data
that are loaded from different data sources.  The ``[]`` operator is
used to access set and parameter values.  Consider the following
example, which loads data and prints the value of the ``[]`` operator:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_getitem.spy
    :language: python

The :class:`DataPortal <ref pyomo.core.base.DataPortal.DataPortal>`
class also has several methods for iterating over the data that has been
loaded:

* ``keys()``: Returns an iterator of the data keys.
* ``values()``: Returns an iterator of the data values.
* ``items()``: Returns an iterator of (name, value) tuples from the
  data.

Finally, the ``data()`` method provides a generic mechanism for
accessing the underlying data representation used by :class:`DataPortal
<ref pyomo.core.base.DataPortal.DataPortal>` objects.

Loading Tabular Data
--------------------

Many data sources supported by Pyomo are tabular data formats.  Tabular
data is numerical or textual data that is organized into one or more
simple tables, where data is arranged in a matrix.  Each table consists
of a matrix of numeric string values, simple strings, and quoted
strings.  All rows have the same length, all columns have the same
length, and the first row typically represents labels for the column
data.

The following section describes the tabular data sources supported by
Pyomo, and the subsequent sections illustrate ways that data can be
loaded from tabular data using TAB files.  Subsequent sections describe
options for loading data from Excel spreadsheets and relational
databases.

Tabular Data
^^^^^^^^^^^^

TAB files represent tabular data in an ascii file using whitespace as a
delimiter.  A TAB file consists of rows of values, where each row has
the same length.  For example, the file ``PP.tab`` has the format:

.. literalinclude:: ../../tests/dataportal/PP.tab
    :language: none

CSV files represent tabular data in a format that is very similar to TAB
files.  Pyomo assumes that a CSV file consists of rows of values, where
each row has the same length.  For example, the file ``PP.csv`` has the
format:

.. literalinclude:: ../../tests/dataportal/PP.csv
    :language: none

Excel spreadsheets can express complex data relationships.  A *range* is
a contiguous, rectangular block of cells in an Excel spreadsheet.  Thus,
a range in a spreadsheet has the same tabular structure as is a TAB file
or a CSV file.  For example, consider the file ``excel.xls`` that has
the range ``PPtable``:

.. image:: PP.png
    :width: 2.5in

A relational database is an application that organizes data into one or
more tables (or *relations*) with a unique key in each row.  Tables both
reflect the data in a database as well as the result of queries within a
database.

XML files represent tabular using ``table`` and ``row`` elements.  Each
sub-element of a ``row`` element represents a different column, where
each row has the same length.  For example, the file ``PP.xml`` has the
format:

.. literalinclude:: ../../tests/dataportal/PP.xml
    :language: none

Loading Set Data
^^^^^^^^^^^^^^^^

The ``set`` option is used specify a ``Set`` component that is loaded
with data.

Loading a Simple Set
""""""""""""""""""""

Consider the file ``A.tab``, which defines a simple set:

.. literalinclude:: ../../tests/dataportal/A.tab
    :language: none

In the following example, a :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object loads data for a simple
set ``A``:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_set1.spy
    :language: python

Loading a Set of Tuples
"""""""""""""""""""""""

Consider the file ``C.tab``:

.. literalinclude:: ../../tests/dataportal/C.tab
    :language: none

In the following example, a :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object loads data for a
two-dimensional set ``C``:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_set2.spy
    :language: python

In this example, the column titles do not directly impact the process of
loading data.  Column titles can be used to select a subset of columns
from a table that is loaded (see below).

Loading a Set Array
"""""""""""""""""""

Consider the file ``D.tab``, which defines an array representation of a
two-dimensional set:

.. literalinclude:: ../../tests/dataportal/D.tab
    :language: none

In the following example, a :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object loads data for a
two-dimensional set ``D``:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_set3.spy
    :language: python

The ``format`` option indicates that the set data is declared in a array
format.

Loading Parameter Data
^^^^^^^^^^^^^^^^^^^^^^

The ``param`` option is used specify a ``Param`` component that is
loaded with data.

Loading a Simple Parameter
""""""""""""""""""""""""""

The simplest parameter is simply a singleton value.  Consider the file
``Z.tab``:

.. literalinclude:: ../../tests/dataportal/Z.tab
    :language: none

In the following example, a :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object loads data for a simple
parameter ``z``:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_param1.spy
    :language: python

Loading an Indexed Parameter
""""""""""""""""""""""""""""

An indexed parameter can be defined by a single column in a table.  For
example, consider the file ``Y.tab``:

.. literalinclude:: ../../tests/dataportal/Y.tab
    :language: none

In the following example, a :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object loads data for an indexed
parameter ``y``:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_param2.spy
    :language: python

When column names are not used to specify the index and parameter data,
then the :class:`DataPortal <ref pyomo.core.base.DataPortal.DataPortal>`
object assumes that the rightmost column defines parameter values.  In
this file, the ``A`` column contains the index values, and the ``Y``
column contains the parameter values.

Loading Set and Parameter Values
""""""""""""""""""""""""""""""""

Note that the data for set ``A`` is predefined in the previous example.
The index set can be loaded with the parameter data using the ``index``
option.  In the following example, a :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object loads data for set ``A``
and the indexed parameter ``y``

.. literalinclude:: ../../tests/dataportal/dataportal_tab_param3.spy
    :language: python

An index set with multiple dimensions can also be loaded with an indexed
parameter.  Consider the file ``PP.tab``:

.. literalinclude:: ../../tests/dataportal/PP.tab
    :language: none

In the following example, a :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object loads data for a tuple
set and an indexed parameter:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_param10.spy
    :language: python

Loading a Parameter with Missing Values
"""""""""""""""""""""""""""""""""""""""

Missing parameter data can be expressed in two ways.  First, parameter
data can be defined with indices that are a subset of valid indices in
the model.  The following example loads the indexed parameter ``y``:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_param9.spy
    :language: python

The model defines an index set with four values, but only three
parameter values are declared in the data file ``Y.tab``.

Parameter data can also be declared with missing values using the period
(``.``) symbol.  For example, consider the file ``S.tab``:

.. literalinclude:: ../../tests/dataportal/PP.tab
    :language: none

In the following example, a :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object loads data for the index
set ``A`` and indexed parameter ``y``:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_param8.spy
    :language: python

The period (``.``) symbol indicates a missing parameter value, but the
index set ``A`` contains the index value for the missing parameter.

Loading Multiple Parameters
"""""""""""""""""""""""""""

Multiple parameters can be initialized at once by specifying a list (or
tuple) of component parameters.  Consider the file ``XW.tab``:

.. literalinclude:: ../../tests/dataportal/XW.tab
    :language: none

In the following example, a :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object loads data for parameters
``x`` and ``w``:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_param4.spy
    :language: python

Selecting Parameter Columns
"""""""""""""""""""""""""""

We have previously noted that the column names do not need to be
specified to load set and parameter data.  However, the ``select``
option can be to identify the columns in the table that are used to load
parameter data.  This option specifies a list (or tuple) of column names
that are used, in that order, to form the table that defines the
component data.

For example, consider the following load declaration:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_param5.spy
    :language: python

The columns ``A`` and ``W`` are selected from the file ``XW.tab``, and a
single parameter is defined.

Loading a Parameter Array
"""""""""""""""""""""""""

Consider the file ``U.tab``, which defines an array representation of a
multiply-indexed parameter:

.. literalinclude:: ../../tests/dataportal/U.tab
    :language: none

In the following example, a :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object loads data for a
two-dimensional parameter ``u``:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_param6.spy
    :language: python

The ``format`` option indicates that the parameter data is declared in a
array format.  The ``format`` option can also indicate that the
parameter data should be transposed.

.. literalinclude:: ../../tests/dataportal/dataportal_tab_param7.spy
    :language: python

Note that the transposed parameter data changes the index set for the
parameter.

Loading from Spreadsheets and Databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tabular data can be loaded from spreadsheets and databases using
auxilliary Python packages that provide an interface to these data
formats.  Data can be loaded from Excel spreadsheets using the
``win32com``, ``xlrd`` and ``openpyxl`` packages.  For example, consider
the following range of cells, which is named ``PPtable``:

.. image:: PP.png
    :width: 2.5in

In the following example, a :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object loads the named range
``PPtable`` from the file ``excel.xls``:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_excel1.spy
    :language: python

Note that the ``range`` option is required to specify the table of cell
data that is loaded from the spreadsheet.

There are a variety of ways that data can be loaded from a relational
database.  In the simplest case, a table can be specified within a
database:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_db1.spy
    :language: python

In this example, the interface ``sqlite3`` is used to load data from an
SQLite database in the file ``PP.sqlite``.  More generally, an SQL query
can be specified to dynamicly generate a table.  For example:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_db2.spy
    :language: python

Data Namespaces
---------------

The :class:`DataPortal <ref pyomo.core.base.DataPortal.DataPortal>`
class supports the concept of a *namespace* to organize data into named
groups that can be enabled or disabled during model construction.
Various :class:`DataPortal <ref pyomo.core.base.DataPortal.DataPortal>`
methods have an optional ``namespace`` argument that defaults to
``None``:

* ``data(name=None, namespace=None)``: Returns the data associated with
  data in the specified namespace

* ``[]``: For a :class:`DataPortal <ref
  pyomo.core.base.DataPortal.DataPortal>` object ``data``, the function
  ``data['A']`` returns data corresponding to ``A`` in the default
  namespace, and ``data['ns1','A']`` returns data corresponding to ``A``
  in namespace ``ns1``.

* ``namespaces()``: Returns an iteratore for the data namespaces.

* ``keys(namespace=None)``: Returns an iterator of the data keys in the
  specified namespace.

* ``values(namespace=None)``: Returns and iterator of the data values in
  the specified namespace.

* ``items(namespace=None)``: Returns an iterator of (name, value) tuples
  in the specified namespace.

By default, data within a namespace are ignored during model
construction.  However, concrete models can be initialized with data
from a specific namespace.  Further, abstract models can be initialized
with a list of namespaces that define the data used to initialized model
components.  For example, the following script generates two model
instances from an abstract model using data loaded into different
namespaces:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_namespaces1.spy
    :language: python


