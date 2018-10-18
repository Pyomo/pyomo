Storing Data from Pyomo Models
==============================

Currently, Pyomo has rather limited capabilities for storing model data
into standard Python data types and serialized data formats.  However,
this capability is under active development.


Serialized Data Formats
-----------------------

JSON and YAML files are structured data formats that are well-suited for
data serialization.  These data formats do not represent data in tabular
format, but instead they directly represent set and parameter values
with lists and dictionaries:

* **Simple Set**: a list of string or numeric value

* **Indexed Set**: a dictionary that maps an index to a list of string
  or numeric value

* **Simple Parameter**: a string or numeric value

* **Indexed Parameter**: a dictionary that maps an index to a numeric
  value

For example, consider the following JSON file:

.. literalinclude:: ../../tests/dataportal/T.json
    :language: none

The data in this file can be used to load the following model:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_json1.spy
    :language: python

Note that no ``set`` or ``param`` option needs to be specified when
loading a ``JSON`` or ``YAML`` file.  All of the set and parameter
data in the file are loaded by the :class:`DataPortal <ref
pyomo.core.base.DataPortal.DataPortal>` object, and only the data
needed for model construction is used.

The following YAML file has a similar structure:

.. literalinclude:: ../../tests/dataportal/T.yaml
    :language: none

The data in this file can be used to load a Pyomo model with the
same syntax as a JSON file:

.. literalinclude:: ../../tests/dataportal/dataportal_tab_yaml1.spy
    :language: python

Storing Model Data in Excel
---------------------------

.. Admonition:: TODO

    More here.

