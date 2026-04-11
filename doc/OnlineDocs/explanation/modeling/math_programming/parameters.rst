Parameters
==========

.. currentmodule:: pyomo.environ
.. doctest::
   :hide:

   >>> import pyomo.environ as pyo
   >>> model = pyo.ConcreteModel()

The word "parameters" is used in many settings. In Pyomo, a :class:`Param` 
represents the fixed data of an optimization model. Unlike variables 
(:class:`Var`), which the solver determines, parameters are inputs that define 
 the specific instance of the problem you are solving.

Common examples of parameters include costs, demands, capacities, or physical 
constants. While you could use standard Python variables to store these values, 
using Pyomo :class:`Param` components offers several advantages:

* **Index Management:** Params can be indexed by Pyomo :class:`Set` objects, 
  ensuring consistency between your data and the model structure. In a 
  :class:`ConcreteModel`, they can also be indexed by standard Python iterables 
  like lists, tuples, or ranges.
* **Validation:** You can define rules to ensure that the data provided (e.g., 
  from an external file) is valid before solving.
* **Symbolic Representation:** In large or complex models, using Params allows 
  Pyomo to maintain the structure of the model separately from the specific values.

.. note::

   When working with a :class:`ConcreteModel`, many modelers choose to use 
   standard Python variables, lists, or dictionaries to store their data 
   instead of Pyomo :class:`Param` objects. This is a common and valid 
   practice. 
   
   However, you must use a Pyomo :class:`Param` if:
   
   * You are using an :class:`AbstractModel` (which requires components to be 
     declared before data is loaded).
   * You need a **mutable** parameter to change values and re-solve the model 
     without the overhead of rebuilding it from scratch.
   * You want to leverage Pyomo's built-in data validation and index-checking 
     capabilities.

Declaration and Options
-----------------------

Parameters are declared as instances of the :class:`Param` class. They can be 
scalar (single value) or indexed by one or more sets (Pyomo :class:`Set` or 
other iterables). For example:

.. testcode::

   model.A = pyo.RangeSet(1,3)
   model.B = pyo.Set(initialize=['dog', 'cat'])
   # Scalar parameter
   model.rho = pyo.Param(initialize=0.5)
   # Indexed parameter (by Set)
   model.P = pyo.Param(model.A, model.B)
   # Indexed parameter (by standard list)
   model.Q = pyo.Param(['a', 'b', 'c'], initialize={'a': 1, 'b': 2, 'c': 3})

If there are indexes for a :class:`Param`, they are provided as the first 
positional arguments and do not have a keyword label. In addition to these 
optional indexes, :class:`Param` takes the following keyword arguments:

- ``default`` = The parameter value used if no other value is specified for an index.
- ``doc`` = A string describing the parameter.
- ``initialize`` = A function, dictionary, or other Python object used to 
  provide initial data.
- ``mutable`` = Boolean indicating if values can be changed after construction 
  (see below).
- ``validate`` = A callback function to verify data integrity.
- ``within`` = A set (e.g., ``NonNegativeReals``) used for domain validation.

Basic Initialization
--------------------

There are many ways to provide data to a :class:`Param`. For example, given 
``model.A`` with values ``{1, 2, 3}``, here are two ways to create a diagonal 
matrix:

.. testcode::

   v={}
   v[1,1] = 9
   v[2,2] = 16
   v[3,3] = 25
   model.S1 = pyo.Param(model.A, model.A, initialize=v, default=0)

You can also use an initialization function that Pyomo calls for each index:

.. testcode::

   def s_init(model, i, j):
       if i == j:
           return i*i
       else:
           return 0.0
   model.S2 = pyo.Param(model.A, model.A, initialize=s_init)

.. note::

   In an :class:`AbstractModel`, data specified in an external input file (e.g., 
   a ``.dat`` file) will override the data specified by the ``initialize`` 
   option.

Validation
----------

Parameter values can be checked by a validation function. In the following 
example, we ensure every value of ``model.T`` is greater than 3.14159:

.. testcode::

   t_data = {1: 10, 2: 3, 3: 20}

   def t_validate(model, v, i):
       return v > 3.14159

   model.T = pyo.Param(model.A, validate=t_validate, initialize=t_data)

This example will produce the following error:

.. testoutput::

   Traceback (most recent call last):
     ...
   ValueError: Invalid parameter value: T[2] = '3', value type=<class 'int'>.
       Value failed parameter validation rule

Performance vs. Flexibility: Mutable Parameters
-----------------------------------------------

By default, Pyomo parameters are **immutable** (``mutable=False``). This choice 
is driven by performance:

* **Immutable (Default):** Pyomo "pre-computes" these values into the algebraic 
  expressions during model construction. This results in faster model generation 
  and significantly lower memory usage, especially for large models. Key 
  advantages include:

  * **Memory Efficiency:** For indexed parameters, Pyomo avoids creating 
    individual component data objects, significantly reducing memory overhead.
  * **Expression Speed:** Values are injected as constants directly into the 
    expression tree. This allows Pyomo to optimize expression tree walking.
  * **Simplification:** Pyomo can simplify constant sub-expressions during 
    model construction (e.g., ``5 * model.p * model.q[i]`` is simplified to a 
    single float if ``p`` and ``q`` are immutable), further accelerating 
    subsequent processing.
* **Mutable:** Pyomo maintains the parameter as a symbolic object within 
  expressions. This allows you to change the value and re-solve without 
  rebuilding the entire model, but it adds computational overhead.

It is important to note that even immutable :class:`Param` objects carry some 
overhead. For the fastest possible model instantiation in a 
:class:`ConcreteModel`, using native Python data structures (like dictionaries 
or lists) to provide values directly into expressions is usually faster than 
using :class:`Param` components. However, as noted earlier, :class:`Param` 
provides benefits like validation and the ability to update values if declared 
as mutable.

When to use Mutable
~~~~~~~~~~~~~~~~~~~

**Use Immutable if:**
  * The data is static and never changes during the lifetime of the model.
  * You want to maximize performance and minimize memory usage for large models.

**Use Mutable if:**
  * You are running a loop (e.g., sensitivity analysis) where you change 
    parameter values and re-solve.
  * You want to update values frequently without the "re-construction" 
    bottleneck.
  * The parameter is part of a nonlinear expression that you need to update.
  * You want named constants to be preserved in the Pyomo expressions (e.g., for documentation of debugging purposes)

Comparison: Param vs. Var
-------------------------

It is common to confuse mutable parameters with variables. The following table 
summarizes the key differences:

.. list-table::
   :header-rows: 1

   * - Feature
     - Param (Immutable)
     - Param (Mutable)
     - Var (fixed)
     - Var (free)
   * - Can change after model construction?
     - No
     - Yes
     - Yes
     - Yes
   * - Rebuilds model on change?
     - Yes (requires new Param)
     - No
     - No
     - No
   * - Solver sees it as:
     - A constant number
     - A constant number
     - A constant number
     - An optimization variable

.. note::

   **Should I use a mutable Param or a fixed Var?**
   While functionally similar, you should use a :class:`Param` for data that 
   defines the problem instance (like costs or demands) and a :class:`Var` for 
   the decisions the solver needs to make. Use `fix()` on a :class:`Var` when 
   you want to temporarily hold a decision constant, and use a mutable 
   :class:`Param` when you need to update input data for sensitivity analysis 
    or iterative algorithms.
