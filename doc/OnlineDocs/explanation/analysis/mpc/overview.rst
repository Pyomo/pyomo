Overview
========

What does this package contain?
-------------------------------

#. Data structures for values and time series data associated with time-indexed variables (or parameters, or named expressions). Examples are setpoint values associated with a subset of state variables or time series data from a simulation

#. Utilities for loading and extracting this data into and from variables in a model

#. Utilities for constructing components from this data (expressions, constraints, and objectives) that are useful for dynamic optimization

What is the goal of this package?
---------------------------------

This package was written to help developers of Pyomo-based dynamic optimization
case studies, especially rolling horizon dynamic optimization case studies,
write scripts that are small, legible, and maintainable.
It does this by providing utilities for mundane data-management and model
construction tasks, allowing the developer to focus on their application.

Why is this package useful?
---------------------------

First, it is not normally easy to extract "flattened" time series data,
in which all indexing structure other than time-indexing has been
flattened to yield a set of one-dimensional arrays, from a Pyomo model.
This is an extremely convenient data structure to have for plotting,
analysis, initialization, and manipulation of dynamic models.
If all variables are indexed by time and only time, this data is relatively
easy to obtain.
The first issue comes up when dealing with components that are indexed by
time in addition to some other set(s). For example:

.. doctest::

   >>> import pyomo.environ as pyo

   >>> m = pyo.ConcreteModel()
   >>> m.time = pyo.Set(initialize=[0, 1, 2])
   >>> m.comp = pyo.Set(initialize=["A", "B"])
   >>> m.var = pyo.Var(m.time, m.comp, initialize=1.0)

   >>> t0 = m.time.first()
   >>> data = {
   ...    m.var[t0, j].name: [m.var[i, j].value for i in m.time]
   ...    for j in m.comp
   ... }
   >>> data
   {'var[0,A]': [1.0, 1.0, 1.0], 'var[0,B]': [1.0, 1.0, 1.0]}

To generate data in this form, we need to (a) know that our variable is indexed
by time and ``m.comp`` and (b) arbitrarily select a time index ``t0`` to
generate a unique key for each time series.
This gets more difficult when blocks and time-indexed blocks are used as well.
The first difficulty can be alleviated using
``flatten_dae_components`` from ``pyomo.dae.flatten``:

.. doctest::

   >>> import pyomo.environ as pyo
   >>> from pyomo.dae.flatten import flatten_dae_components

   >>> m = pyo.ConcreteModel()
   >>> m.time = pyo.Set(initialize=[0, 1, 2])
   >>> m.comp = pyo.Set(initialize=["A", "B"])
   >>> m.var = pyo.Var(m.time, m.comp, initialize=1.0)

   >>> t0 = m.time.first()
   >>> scalar_vars, dae_vars = flatten_dae_components(m, m.time, pyo.Var)
   >>> data = {var[t0].name: list(var[:].value) for var in dae_vars}
   >>> data
   {'var[0,A]': [1.0, 1.0, 1.0], 'var[0,B]': [1.0, 1.0, 1.0]}

Addressing the arbitrary ``t0`` index requires us to ask what key we
would like to use to identify each time series in our data structure.
The key should uniquely correspond to a component, or "sub-component"
that is indexed only by time. A slice, e.g. ``m.var[:, "A"]`` seems
natural. However, Pyomo provides a better data structure that can
be constructed from a component, slice, or string, called
``ComponentUID``. Being constructable from a string is important as
we may want to store or serialize this data in a form that is agnostic
of any particular ``ConcreteModel`` object.
We can now generate our data structure as:

.. doctest::

   >>> data = {
   ...    pyo.ComponentUID(var.referent): list(var[:].value)
   ...    for var in dae_vars
   ... }
   >>> data
   {var[*,A]: [1.0, 1.0, 1.0], var[*,B]: [1.0, 1.0, 1.0]}

This is the structure of the underlying dictionary in the ``TimeSeriesData``
class provided by this package. We can generate this data using this package
as:

.. doctest::

   >>> import pyomo.environ as pyo
   >>> from pyomo.contrib.mpc import DynamicModelInterface

   >>> m = pyo.ConcreteModel()
   >>> m.time = pyo.Set(initialize=[0, 1, 2])
   >>> m.comp = pyo.Set(initialize=["A", "B"])
   >>> m.var = pyo.Var(m.time, m.comp, initialize=1.0)

   >>> # Construct a helper class for interfacing model with data
   >>> helper = DynamicModelInterface(m, m.time)

   >>> # Generates a TimeSeriesData object
   >>> series_data = helper.get_data_at_time()

   >>> # Get the underlying dictionary
   >>> data = series_data.get_data()
   >>> data
   {var[*,A]: [1.0, 1.0, 1.0], var[*,B]: [1.0, 1.0, 1.0]}

The first value proposition of this package is that ``DynamicModelInterface``
and ``TimeSeriesData`` provide wrappers to ease loading and extraction of data
via ``flatten_dae_components`` and ``ComponentUID``.

The second difficulty addressed by this package is that of extracting and
loading data between (potentially) different models.
For instance, in model predictive control, we often want to extract data from
a particular time point in a plant model and load it into a controller model
as initial conditions. This can be done as follows:

.. doctest::

   >>> import pyomo.environ as pyo
   >>> from pyomo.contrib.mpc import DynamicModelInterface

   >>> m1 = pyo.ConcreteModel()
   >>> m1.time = pyo.Set(initialize=[0, 1, 2])
   >>> m1.comp = pyo.Set(initialize=["A", "B"])
   >>> m1.var = pyo.Var(m1.time, m1.comp, initialize=1.0)

   >>> m2 = pyo.ConcreteModel()
   >>> m2.time = pyo.Set(initialize=[0, 1, 2])
   >>> m2.comp = pyo.Set(initialize=["A", "B"])
   >>> m2.var = pyo.Var(m2.time, m2.comp, initialize=2.0)

   >>> # Construct helper objects
   >>> m1_helper = DynamicModelInterface(m1, m1.time)
   >>> m2_helper = DynamicModelInterface(m2, m2.time)

   >>> # Extract data from final time point of m2
   >>> tf = m2.time.last()
   >>> tf_data = m2_helper.get_data_at_time(tf)

   >>> # Load data into initial time point of m1
   >>> t0 = m1.time.first()
   >>> m1_helper.load_data(tf_data, time_points=t0)

   >>> # Get TimeSeriesData object
   >>> series_data = m1_helper.get_data_at_time()
   >>> # Get underlying dictionary
   >>> series_data.get_data()
   {var[*,A]: [2.0, 1.0, 1.0], var[*,B]: [2.0, 1.0, 1.0]}

.. note::

   Here we rely on the fact that our variable has the same name in
   both models.

Finally, this package provides methods for constructing components like
tracking cost expressions and piecewise-constant constraints from the
provided data structures. For example, the following code constructs
a tracking cost expression.

.. doctest::

   >>> import pyomo.environ as pyo
   >>> from pyomo.contrib.mpc import DynamicModelInterface

   >>> m = pyo.ConcreteModel()
   >>> m.time = pyo.Set(initialize=[0, 1, 2])
   >>> m.comp = pyo.Set(initialize=["A", "B"])
   >>> m.var = pyo.Var(m.time, m.comp, initialize=1.0)

   >>> # Construct helper object
   >>> helper = DynamicModelInterface(m, m.time)

   >>> # Construct data structure for setpoints
   >>> setpoint = {m.var[:, "A"]: 0.5, m.var[:, "B"]: 2.0}
   >>> var_set, tr_cost = helper.get_penalty_from_target(setpoint)
   >>> m.setpoint_idx = var_set
   >>> m.tracking_cost = tr_cost
   >>> m.tracking_cost.pprint()
   tracking_cost : Size=6, Index=setpoint_idx*time
       Key    : Expression
       (0, 0) : (var[0,A] - 0.5)**2
       (0, 1) : (var[1,A] - 0.5)**2
       (0, 2) : (var[2,A] - 0.5)**2
       (1, 0) : (var[0,B] - 2.0)**2
       (1, 1) : (var[1,B] - 2.0)**2
       (1, 2) : (var[2,B] - 2.0)**2


These methods will hopefully allow developers to declutter dynamic optimization
scripts and pay more attention to the application of the optimization problem
rather than the setup of the optimization problem.

Who develops and maintains this package?
----------------------------------------

This package was developed by Robert Parker while a PhD student in Larry
Biegler's group at CMU, with guidance from Bethany Nicholson and John Siirola.
