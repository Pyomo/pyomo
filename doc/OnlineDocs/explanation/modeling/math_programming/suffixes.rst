.. _Suffixes:

Suffixes
========

Suffixes provide a mechanism for declaring extraneous model data, which
can be used in a number of contexts. Most commonly, suffixes are used by
solver plugins to store extra information about the solution of a
model. This and other suffix functionality is made available to the
modeler through the use of the Suffix component class.  Uses of Suffix
include:

- Importing extra information from a solver about the solution of a
  mathematical program (e.g., constraint duals, variable reduced costs,
  basis information).
- Exporting information to a solver or algorithm to aid in solving a
  mathematical program (e.g., warm-starting information, variable
  branching priorities).
- Tagging modeling components with local data for later use in advanced
  scripting algorithms.

Suffix Notation and the Pyomo NL File Interface
-----------------------------------------------

The Suffix component used in Pyomo has been adapted from the suffix
notation used in the modeling language AMPL [FGK02]_. Therefore, it
follows naturally that AMPL style suffix functionality is fully
available using Pyomo's NL file interface. For information on AMPL style
suffixes the reader is referred to the AMPL website:

    http://www.ampl.com

A number of scripting examples that highlight the use AMPL style suffix
functionality are available in the ``examples/pyomo/suffixes`` directory
distributed with Pyomo.

Declaration
-----------

The effects of declaring a Suffix component on a Pyomo model are
determined by the following traits:

- direction: This trait defines the direction of information flow for
  the suffix. A suffix direction can be assigned one of four possible
  values:

  + ``LOCAL`` - suffix data stays local to the modeling framework and
    will not be imported or exported by a solver plugin (default)
  + ``IMPORT`` - suffix data will be imported from the solver by its
    respective solver plugin
  + ``EXPORT`` - suffix data will be exported to a solver by its
    respective solver plugin
  + ``IMPORT_EXPORT`` - suffix data flows in both directions between the
    model and the solver or algorithm

- datatype: This trait advertises the type of data held on the suffix
  for those interfaces where it matters (e.g., the NL file interface). A
  suffix datatype can be assigned one of three possible values:

  + ``FLOAT`` - the suffix stores floating point data (default)
  + ``INT`` - the suffix stores integer data
  + ``None`` - the suffix stores any type of data

.. note::

   Exporting suffix data through Pyomo's NL file interface requires all
   active export suffixes have a strict datatype (i.e.,
   ``datatype=None`` is not allowed).

The following code snippet shows examples of declaring a Suffix
component on a Pyomo model:

.. testcode::

   import pyomo.environ as pyo

   model = pyo.ConcreteModel()

   # Export integer data
   model.priority = pyo.Suffix(
       direction=pyo.Suffix.EXPORT, datatype=pyo.Suffix.INT)

   # Export and import floating point data
   model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

   # Store floating point data
   model.junk = pyo.Suffix()

Declaring a Suffix with a non-local direction on a model is not
guaranteed to be compatible with all solver plugins in Pyomo. Whether a
given Suffix is acceptable or not depends on both the solver and solver
interface being used. In some cases, a solver plugin will raise an
exception if it encounters a Suffix type that it does not handle, but
this is not true in every situation. For instance, the NL file interface
is generic to all AMPL-compatible solvers, so there is no way to
validate that a Suffix of a given name, direction, and datatype is
appropriate for a solver. One should be careful in verifying that Suffix
declarations are being handled as expected when switching to a different
solver or solver interface.

Operations
----------

The Suffix component class provides a dictionary interface for mapping
Pyomo modeling components to arbitrary data. This mapping functionality
is captured within the ComponentMap base class, which is also available
within Pyomo's modeling environment. The ComponentMap can be used as a
more lightweight replacement for Suffix in cases where a simple mapping
from Pyomo modeling components to arbitrary data values is required.

.. note::

   ComponentMap and Suffix use the built-in ``id()`` function for
   hashing entry keys. This design decision arises from the fact that
   most of the modeling components found in Pyomo are either not
   hashable or use a hash based on a mutable numeric value, making them
   unacceptable for use as keys with the built-in ``dict`` class.

.. warning::

   The use of the built-in ``id()`` function for hashing entry keys in
   ComponentMap and Suffix makes them inappropriate for use in
   situations where built-in object types must be used as keys. It is
   strongly recommended that only Pyomo modeling components be used as
   keys in these mapping containers (``Var``, ``Constraint``, etc.).

.. warning::

   Do not attempt to pickle or deepcopy instances of ComponentMap or
   Suffix unless doing so along with the components for which they hold
   mapping entries. As an example, placing one of these objects on a
   model and then cloning or pickling that model is an acceptable
   scenario.

In addition to the dictionary interface provided through the
ComponentMap base class, the Suffix component class also provides a
number of methods whose default semantics are more convenient for
working with indexed modeling components.  The easiest way to highlight
this functionality is through the use of an example.

.. testcode::

   model = pyo.ConcreteModel()
   model.x = pyo.Var()
   model.y = pyo.Var([1,2,3])
   model.foo = pyo.Suffix()

In this example we have a concrete Pyomo model with two different types
of variable components (indexed and non-indexed) as well as a Suffix
declaration (foo). The next code snippet shows examples of adding
entries to the suffix foo.

.. testcode::

   # Assign a suffix value of 1.0 to model.x
   model.foo.set_value(model.x, 1.0)

   # Same as above with dict interface
   model.foo[model.x] = 1.0

   # Assign a suffix value of 0.0 to all indices of model.y
   # By default this expands so that entries are created for
   # every index (y[1], y[2], y[3]) and not model.y itself
   model.foo.set_value(model.y, 0.0)

   # The same operation using the dict interface results in an entry only
   # for the parent component model.y
   model.foo[model.y] = 50.0

   # Assign a suffix value of -1.0 to model.y[1]
   model.foo.set_value(model.y[1], -1.0)

   # Same as above with the dict interface
   model.foo[model.y[1]] = -1.0


In this example we highlight the fact that the ``__setitem__`` and
``setValue`` entry methods can be used interchangeably except in the
case where indexed components are used (model.y). In the indexed case,
the ``__setitem__`` approach creates a single entry for the parent
indexed component itself, whereas the ``setValue`` approach by default
creates an entry for each index of the component. This behavior can be
controlled using the optional keyword 'expand', where assigning it a
value of ``False`` results in the same behavior as ``__setitem__``.

Other operations like accessing or removing entries in our mapping can
performed as if the built-in ``dict`` class is in use.

.. doctest::

   >>> print(model.foo.get(model.x))
   1.0
   >>> print(model.foo[model.x])
   1.0

   >>> print(model.foo.get(model.y[1]))
   -1.0
   >>> print(model.foo[model.y[1]])
   -1.0

   >>> print(model.foo.get(model.y[2]))
   0.0
   >>> print(model.foo[model.y[2]])
   0.0

   >>> print(model.foo.get(model.y))
   50.0
   >>> print(model.foo[model.y])
   50.0

   >>> del model.foo[model.y]
   >>> print(model.foo.get(model.y))
   None

   >>> print(model.foo[model.y])
   Traceback (most recent call last):
     ...
   KeyError: "Component with id '...': y"


The non-dict method ``clear_value`` can be used in place of
``__delitem__`` to remove entries, where it inherits the same default
behavior as ``setValue`` for indexed components and does not raise a
KeyError when the argument does not exist as a key in the mapping.

.. doctest::

   >>> model.foo.clear_value(model.y)

   >>> print(model.foo[model.y[1]])
   Traceback (most recent call last):
     ...
   KeyError: "Component with id '...': y[1]"

   >>> del model.foo[model.y[1]]
   Traceback (most recent call last):
     ...
   KeyError: "Component with id '...': y[1]"

   >>> model.foo.clear_value(model.y[1])

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

Importing suffix information from a solver solution is achieved by
declaring a Suffix component with the appropriate name and
direction. Suffix names available for import may be specific to
third-party solvers as well as individual solver interfaces within
Pyomo. The most common of these, available with most solvers and solver
interfaces, is constraint dual multipliers. Requesting that duals be
imported into suffix data can be accomplished by declaring a Suffix
component on the model.

.. testcode::

   model = pyo.ConcreteModel()
   model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
   model.x = pyo.Var()
   model.obj = pyo.Objective(expr=model.x)
   model.con = pyo.Constraint(expr=model.x >= 1.0)

The existence of an active suffix with the name dual that has an import
style suffix direction will cause constraint dual information to be
collected into the solver results (assuming the solver supplies dual
information). In addition to this, after loading solver results into a
problem instance (using a python script or Pyomo callback functions in
conjunction with the ``pyomo`` command), one can access the dual values
associated with constraints using the dual Suffix component.

.. doctest::
   :skipif: not glpk_available

   >>> results = pyo.SolverFactory('glpk').solve(model)
   >>> pyo.assert_optimal_termination(results)
   >>> print(model.dual[model.con])
   1.0

Alternatively, the ``pyomo`` option ``--solver-suffixes`` can be used to
request suffix information from a solver. In the event that suffix names
are provided via this command-line option, the ``pyomo`` script will
automatically declare these Suffix components on the constructed
instance making these suffixes available for import.

Exporting Suffix Data
---------------------

Exporting suffix data is accomplished in a similar manner as to that of
importing suffix data. One simply needs to declare a Suffix component on
the model with an export style suffix direction and associate modeling
component values with it. The following example shows how one can
declare a special ordered set of type 1 using AMPL-style suffix notation
in conjunction with Pyomo's NL file interface.

.. testcode::

   model = pyo.ConcreteModel()
   model.y = pyo.Var([1,2,3], within=pyo.NonNegativeReals)

   model.sosno = pyo.Suffix(direction=pyo.Suffix.EXPORT)
   model.ref = pyo.Suffix(direction=pyo.Suffix.EXPORT)

   # Add entry for each index of model.y
   model.sosno.set_value(model.y, 1)
   model.ref[model.y[1]] = 0
   model.ref[model.y[2]] = 1
   model.ref[model.y[3]] = 2

Most AMPL-compatible solvers will recognize the suffix names ``sosno``
and ``ref`` as declaring a special ordered set, where a positive value
for ``sosno`` indicates a special ordered set of type 1 and a negative
value indicates a special ordered set of type 2.

.. note::

   Pyomo provides the :class:`SOSConstraint` component for declaring special
   ordered sets, which is recognized by all solver interfaces, including
   the NL file interface.

Pyomo's NL file interface will recognize an EXPORT style Suffix
component with the name 'dual' as supplying initializations for
constraint multipliers. As such it will be treated separately than all
other EXPORT style suffixes encountered in the NL writer, which are
treated as AMPL-style suffixes. The following example script shows how
one can warmstart the interior-point solver Ipopt by supplying both
primal (variable values) and dual (suffixes) solution information. This
dual suffix information can be both imported and exported using a single
Suffix component with an IMPORT_EXPORT direction.

.. testcode::

   model = pyo.ConcreteModel()
   model.x1 = pyo.Var(bounds=(1,5),initialize=1.0)
   model.x2 = pyo.Var(bounds=(1,5),initialize=5.0)
   model.x3 = pyo.Var(bounds=(1,5),initialize=5.0)
   model.x4 = pyo.Var(bounds=(1,5),initialize=1.0)
   model.obj = pyo.Objective(
       expr=model.x1*model.x4*(model.x1 + model.x2 + model.x3) + model.x3)
   model.inequality = pyo.Constraint(
       expr=model.x1*model.x2*model.x3*model.x4 >= 25.0)
   model.equality = pyo.Constraint(
       expr=model.x1**2 + model.x2**2 + model.x3**2 + model.x4**2 == 40.0)

   ### Declare all suffixes
   # Ipopt bound multipliers (obtained from solution)
   model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
   model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
   # Ipopt bound multipliers (sent to solver)
   model.ipopt_zL_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
   model.ipopt_zU_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
   # Obtain dual solutions from first solve and send to warm start
   model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

   ipopt = pyo.SolverFactory('ipopt')


The difference in performance can be seen by examining Ipopt's iteration
log with and without warm starting:

- Without Warmstart:

  .. testcode::
   :skipif: not ipopt_available

   ipopt.solve(model, tee=True)

  .. testoutput::
   :skipif: not ipopt_available

   ...
   iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      0  1.6109693e+01 1.12e+01 5.28e-01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
      1  1.6982239e+01 7.30e-01 1.02e+01  -1.0 6.11e-01    -  7.19e-02 1.00e+00f  1
      2  1.7318411e+01 ...
      ...
      8  1.7014017e+01 ...

   Number of Iterations....: 8
   ...

- With Warmstart:

  .. testcode::
   :skipif: not ipopt_available

   ### Set Ipopt options for warm-start
   # The current values on the ipopt_zU_out and ipopt_zL_out suffixes will
   # be used as initial conditions for the bound multipliers to solve the
   # new problem
   model.ipopt_zL_in.update(model.ipopt_zL_out)
   model.ipopt_zU_in.update(model.ipopt_zU_out)
   ipopt.options['warm_start_init_point'] = 'yes'
   ipopt.options['warm_start_bound_push'] = 1e-6
   ipopt.options['warm_start_mult_bound_push'] = 1e-6
   ipopt.options['mu_init'] = 1e-6

   ipopt.solve(model, tee=True)

  .. testoutput::
   :skipif: not ipopt_available

   ...
   iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      0  1.7014032e+01 2.00e-06 4.07e-06  -6.0 0.00e+00    -  0.00e+00 0.00e+00   0
      1  1.7014019e+01 3.65e-12 1.00e-11  -6.0 2.50e-01    -  1.00e+00 1.00e+00h  1
      2  1.7014017e+01 ...

   Number of Iterations....: 2
   ...

Using Suffixes With an AbstractModel
------------------------------------

In order to allow the declaration of suffix data within the framework of
an AbstractModel, the Suffix component can be initialized with an
optional construction rule. As with constraint rules, this function will
be executed at the time of model construction. The following simple
example highlights the use of the ``rule`` keyword in suffix
initialization. Suffix rules are expected to return an iterable of
(component, value) tuples, where the ``expand=True`` semantics are
applied for indexed components.

.. testcode::

   model = pyo.AbstractModel()
   model.x = pyo.Var()
   model.c = pyo.Constraint(expr=model.x >= 1)

   def foo_rule(m):
      return ((m.x, 2.0), (m.c, 3.0))
   model.foo = pyo.Suffix(rule=foo_rule)

.. doctest::

   >>> # Instantiate the model
   >>> inst = model.create_instance()

   >>> print(inst.foo[inst.x])
   2.0
   >>> print(inst.foo[inst.c])
   3.0

   >>> # Note that model.x and inst.x are not the same object
   >>> print(inst.foo[model.x])
   Traceback (most recent call last):
     ...
   KeyError: "Component with id '...': x"

The next example shows an abstract model where suffixes are attached
only to the variables:

.. testcode::

   model = pyo.AbstractModel()
   model.I = pyo.RangeSet(1,4)
   model.x = pyo.Var(model.I)
   def c_rule(m, i):
       return m.x[i] >= i
   model.c = pyo.Constraint(model.I, rule=c_rule)

   def foo_rule(m):
       return ((m.x[i], 3.0*i) for i in m.I)
   model.foo = pyo.Suffix(rule=foo_rule)

.. doctest::

   >>> # instantiate the model
   >>> inst = model.create_instance()
   >>> for i in inst.I:
   ...     print((i, inst.foo[inst.x[i]]))
   (1, 3.0)
   (2, 6.0)
   (3, 9.0)
   (4, 12.0)
