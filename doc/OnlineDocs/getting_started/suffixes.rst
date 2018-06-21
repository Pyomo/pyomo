.. _Suffixes:

Suffixes
========


Suffixes provide a mechanism for declaring extraneous model data,
which can be used in a number of contexts. Most commonly, suffixes are
used by solver plugins to store extra information about the solution
of a model. This and other suffix functionality is made available to
the modeler through the use of the Suffix component class.  Uses of
Suffix include:

- Importing extra information from a solver about the solution of a mathematical program (e.g., constraint duals, variable reduced costs, basis information).
- Exporting information to a solver or algorithm to aid in solving a mathematical program (e.g., warm-starting information, variable branching priorities).
- Tagging modeling components with local data for later use in advanced scripting algorithms.

Suffix Notation and the Pyomo NL File Interface
-----------------------------------------------

The Suffix component used in Pyomo has been adapted from the
suffix notation used in the modeling language AMPL <<AMPL>>. Therefore,
it follows naturally that AMPL style suffix functionality is fully available
using Pyomo's NL file interface. For information on AMPL style suffixes the reader
is referred to the AMPL website:

    http://www.ampl.com

A number of scripting examples that highlight the use AMPL style
suffix functionality are available in the ``examples/pyomo/suffixes``
directory distributed with Pyomo.

Declaration
-----------

The effects of declaring a Suffix component on a Pyomo model are
determined by the following traits:

- direction: This trait defines the direction of information flow for the suffix. A suffix direction can be assigned one of four possible values:

  + ``LOCAL`` - suffix data stays local to the modeling framework and will not be imported or exported by a solver plugin (default)
  + ``IMPORT`` - suffix data will be imported from the solver by its respective solver plugin
  + ``EXPORT`` - suffix data will be exported to a solver by its respective solver plugin
  + ``IMPORT_EXPORT`` - suffix data flows in both directions between the model and the solver or algorithm

- datatype: This trait advertises the type of data held on the suffix for those interfaces where it matters (e.g., the NL file interface). A suffix datatype can be assigned one of three possible values:

  + ``FLOAT`` - the suffix stores floating point data (default)
  + ``INT`` - the suffix stores integer data
  + ``None`` - the suffix stores any type of data

NOTE: Exporting suffix data through Pyomo's NL file interface requires all active export suffixes
have a strict datatype (i.e., ``datatype=None`` is not allowed).

The following code snippet shows examples of declaring a Suffix component on a Pyomo model:

.. literalinclude:: spyfiles/spy4suffixes_Declare_suffix_component.spy
   :language: python

Declaring a Suffix with a non-local direction on a model is not
guaranteed to be compatible with all solver plugins in Pyomo. Whether
a given Suffix is acceptable or not depends on both the solver and
solver interface being used. In some cases, a solver plugin will raise
an exception if it encounters a Suffix type that it does not handle,
but this is not true in every situation. For instance, the NL file
interface is generic to all AMPL-compatible solvers, so there is no way
to validate that a Suffix of a given name, direction, and datatype is
appropriate for a solver. One should be careful in verifying that
Suffix declarations are being handled as expected when switching to
a different solver or solver interface.

Operations
----------

The Suffix component class provides a dictionary interface for mapping
Pyomo modeling components to arbitrary data. This mapping
functionality is captured within the ComponentMap base class, which is
also available within Pyomo's modeling environment. The ComponentMap
can be used as a more lightweight replacement for Suffix in cases
where a simple mapping from Pyomo modeling components to arbitrary
data values is required.

NOTE: ComponentMap and Suffix use the built-in ``id()``
function for hashing entry keys. This design decision
arises from the fact that most of the modeling components
found in Pyomo are either not hashable or use a hash based on a mutable
numeric value, making them unacceptable for use as keys with the
built-in ``dict`` class.

WARNING: The use of the built-in ``id()`` function for hashing entry
keys in ComponentMap and Suffix makes them inappropriate for use in
situations where built-in object types must be used as keys. It is
strongly recommended that only Pyomo modeling components be used as
keys in these mapping containers (``Var``, ``Constraint``, etc.).

WARNING: Do not attempt to pickle or deepcopy instances of
ComponentMap or Suffix unless doing so along with the components for
which they hold mapping entries. As an example, placing one of these
objects on a model and then cloning or pickling that model is an
acceptable scenario.

In addition to the dictionary interface provided through the
ComponentMap base class, the Suffix component class also provides a
number of methods whose default semantics are more convenient for
working with indexed modeling components.  The easiest way to
highlight this functionality is through the use of an example.

.. literalinclude:: spyfiles/spy4suffixes_Use_suffix_component_class_methods.spy
   :language: python

In this example we have a concrete Pyomo model with two different
types of variable components (indexed and non-indexed) as well as a
Suffix declaration (foo). The next code snippet shows examples of adding
entries to the suffix foo.

.. literalinclude:: spyfiles/spy4suffixes_Add_entries_to_suffix_declaration.spy
   :language: python

In this example we highlight the fact that the ``__setitem__`` and
``setValue`` entry methods can be used interchangeably except in the case
where indexed components are used (model.y). In the indexed case, the
``__setitem__`` approach creates a single entry for the parent indexed
component itself, whereas the ``setValue`` approach by default creates an
entry for each index of the component. This behavior can be controlled using
the optional keyword 'expand', where assigning it a value of ``False`` results in the
same behavior as ``__setitem__``.

Other operations like accessing or removing entries in our mapping can performed as if the built-in
``dict`` class is in use.

.. literalinclude:: spyfiles/spy4suffixes_Print_value.spy
   :language: python


The non-dict method ``clear_value`` can be used in place of ``__delitem__`` to
remove entries, where it inherits the same default behavior as
``setValue`` for indexed components and does not raise a KeyError when
the argument does not exist as a key in the mapping.

.. literalinclude:: spyfiles/spy4suffixes_Clear_value.spy
   :language: python


A summary non-dict Suffix methods is provided here:

 |  clearAllValues()
 |      Clears all suffix data.
 |
 |  clear_value(component, expand=True)
 |      Clears suffix information for a component.
 |
 |  setAllValues(value)
 |      Sets the value of this suffix on all components.
 |
 |  setValue(component, value, expand=True)
 |      Sets the value of this suffix on the specified component.
 |
 |  updateValues(data_buffer, expand=True)
 |      Updates the suffix data given a list of component,value tuples. Provides
 |      an improvement in efficiency over calling setValue on every component.
 |
 |  getDatatype()
 |      Return the suffix datatype.
 |
 |  setDatatype(datatype)
 |      Set the suffix datatype.
 |
 |  getDirection()
 |      Return the suffix direction.
 |
 |  setDirection(direction)
 |      Set the suffix direction.
 |
 |  importEnabled()
 |      Returns True when this suffix is enabled for import from solutions.
 |
 |  exportEnabled()
 |      Returns True when this suffix is enabled for export to solvers.

Importing Suffix Data
---------------------

Importing suffix information from a solver solution is achieved by declaring a Suffix component with the appropriate name and direction. Suffix names available for import may be specific to third-party solvers as well as individual solver interfaces within Pyomo. The most common of these, available with most solvers and solver interfaces, is constraint dual multipliers. Requesting that duals be imported into suffix data can be accomplished by declaring a Suffix component on the model.

.. literalinclude:: spyfiles/spy4suffixes_Import_suffix_information.spy
   :language: python

The existence of an active suffix with the name dual that has an import style suffix direction will cause constraint dual information to be collected into the solver results (assuming the solver supplies dual information). In addition to this, after loading solver results into a problem instance (using a python script or Pyomo callback functions in conjunction with the ``pyomo`` command), one can access the dual values associated with constraints using the dual Suffix component.

.. literalinclude:: spyfiles/spy4suffixes_Print_dual_value.spy
   :language: python

Alternatively, the ``pyomo`` option ``--solver-suffixes`` can be used to request suffix information from a solver. In the event that suffix names are provided via this command-line option, the ``pyomo`` script will automatically declare these Suffix components on the constructed instance making these suffixes available for import.

Exporting Suffix Data
---------------------

Exporting suffix data is accomplished in a similar manner as to that
of importing suffix data. One simply needs to declare a Suffix
component on the model with an export style suffix direction and
associate modeling component values with it. The following example
shows how one can declare a special ordered set of type 1 using
AMPL-style suffix notation in conjunction with Pyomo's NL file
interface.

.. literalinclude:: spyfiles/spy4suffixes_Export_suffix_data.spy
   :language: python

Most AMPL-compatible solvers will recognize the suffix names ``sosno``
and ``ref`` as declaring a special ordered set, where a positive value
for ``sosno`` indicates a special ordered set of type 1 and a negative
value indicates a special ordered set of type 2.

NOTE: Pyomo provides the SOSConstraint component for declaring special
ordered sets, which is recognized by all solver interface, including
the NL file interface.

Pyomo's NL file interface will recognize an EXPORT style Suffix
component with the name 'dual' as supplying initializations for
constraint multipliers. As such it will be treated separately than all
other EXPORT style suffixes encountered in the NL writer, which are
treated as AMPL-style suffixes. The following example script shows how
one can warmstart the interior-point solver Ipopt by supplying both
primal (variable values) and dual (suffixes) solution
information. This dual suffix information can be both imported and
exported using a single Suffix component with an IMPORT_EXPORT
direction.

.. literalinclude:: scripts_examples/ipopt_warmstart.py
   :language: python

The difference in performance can be seen by examining Ipopt's
iteration log with and without warm starting:

- Without Warmstart:

::

  iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
    0  1.6109693e+01 1.12e+01 5.28e-01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
    1  1.6982239e+01 7.30e-01 1.02e+01  -1.0 6.11e-01    -  7.19e-02 1.00e+00f  1
    2  1.7318411e+01 3.60e-02 5.05e-01  -1.0 1.61e-01    -  1.00e+00 1.00e+00h  1
    3  1.6849424e+01 2.78e-01 6.68e-02  -1.7 2.85e-01    -  7.94e-01 1.00e+00h  1
    4  1.7051199e+01 4.71e-03 2.78e-03  -1.7 6.06e-02    -  1.00e+00 1.00e+00h  1
    5  1.7011979e+01 7.19e-03 8.50e-03  -3.8 3.66e-02    -  9.45e-01 9.98e-01h  1
    6  1.7014271e+01 1.74e-05 9.78e-06  -3.8 3.33e-03    -  1.00e+00 1.00e+00h  1
    7  1.7014021e+01 1.23e-07 1.82e-07  -5.7 2.69e-04    -  1.00e+00 1.00e+00h  1
    8  1.7014017e+01 1.77e-11 2.52e-11  -8.6 3.32e-06    -  1.00e+00 1.00e+00h  1

    Number of Iterations....: 8

- With Warmstart:

::

  iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
    0  1.7014032e+01 2.00e-06 4.07e-06  -6.0 0.00e+00    -  0.00e+00 0.00e+00   0
    1  1.7014019e+01 3.65e-12 1.00e-11  -6.0 2.50e-01    -  1.00e+00 1.00e+00h  1
    2  1.7014017e+01 4.48e-12 6.43e-12  -9.0 1.92e-06    -  1.00e+00 1.00e+00h  1

    Number of Iterations....: 2

Using Suffixes With an AbstractModel
------------------------------------

In order to allow the declaration of suffix data within the framework
of an AbstractModel, the Suffix component can be initialized with an
optional construction rule. As with constraint rules, this function
will be executed at the time of model construction. The following
simple example highlights the use of the ``rule`` keyword in suffix
initialization. Suffix rules are expected to return an iterable of
(component, value) tuples, where the ``expand=True`` semantics are
applied for indexed components.

.. literalinclude:: spyfiles/spy4suffixes_Suffix_initialization_rule_keyword.spy
   :language: python


The next example shows an abstract model where suffixes are attached
only to the variables:

.. literalinclude:: scripts_examples/AbstractSuffixes.py
   :language: python
