Tutorial
========

Overview
--------

Pyomo includes a diverse set of optimization capabilities for
formulating and analyzing optimization models.  Pyomo supports the
formulation and analysis of mathematical models for complex
optimization applications. This capability is commonly associated
with algebraic modeling languages (AMLs), which support the description
and analysis of mathematical models with a high-level language.
Although most AMLs are implemented in custom modeling languages,
Pyomo’s modeling objects are embedded within Python, a full-featured
high-level programming language that contains a rich set of supporting
libraries.

Pyomo has also proven an effective framework for developing high-level
optimization and analysis tools.  It is easy to develop Python
scripts that use Pyomo as a part of a complex analysis workflow.
Additionally, Pyomo includes a variety of optimization solvers for
stochastic programming, dynamic optimization with differential
algebraic equations, mathematical programming with equilibrium
conditions, and more!  Increasingly, Pyomo is integrating functionality
that is normally associated with an optimization solver library.

Concrete vs Abstract Models
---------------------------

Modeling Components
-------------------
The primary modeling components defined by Pyomo are
:class:`Set<pyomo.core.base.sets.Set>`, :class:`Var<pyomo.core.base.var.Var>`,
:class:`Constraint<pyomo.core.base.constraint.Constraint>`,
:class:`Objective<pyomo.core.base.objective.Objective>`, and
:class:`Block<pyomo.core.base.block.Block>`. Each of these may be indexed.

.. doctest::
   
   >>> import pyomo.envrion as pe
   >>> m = pe.ConcreteModel()
   >>> m.a = pe.Set(initialize=[1, 2, 3])
   >>> m.x = pe.Var(m.a, initialize=0, bounds=(-10,10))
   >>> m.y = pe.Var(m.a)
   >>> def c_rule(m, i):
   >>>    return m.x[i] >= m.y[i]
   >>> m.c = pe.Constraint(m.a, rule=c_rule)
   >>> m.c.pprint()
   c : Size=3, Index=a, Active=True
       Key : Lower : Body        : Upper : Active
         1 :  -Inf : y[1] - x[1] :   0.0 :   True
	 2 :  -Inf : y[2] - x[2] :   0.0 :   True
	 3 :  -Inf : y[3] - x[3] :   0.0 :   True

By default, the index set is fixed. New indices may not be added:

.. doctest::
   
   >>> m.z = pe.Var()
   >>> m.c[4] = m.x[1] == 5 * m.z
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
     File "/Users/michaelbynum/Documents/pyomo_dev/pyomo/core/base/indexed_component.py", line 554, in __setitem__
       index = self._validate_index(index)
     File "/Users/michaelbynum/Documents/pyomo_dev/pyomo/core/base/indexed_component.py", line 649, in _validate_index
       % ( idx, self.name, ))
   KeyError: "Error acccessing indexed component value:\n\nIndex '4' is not valid for indexed component 'c'"

This helps prevent many common mistakes. However, IndexedComponent can behave
more like a dictionary if Any is used as the index.

.. doctest::
   
   >>> m.c2 = pe.Constraint(pe.Any)
   >>> m.c2[1] = m.x[1] == 5 * m.z
   >>> m.c2[8] = m.x[2] == m.z * m.y[2]
   >>> m.c2.pprint()
   c2 : Size=2, Index=Any, Active=True
       Key : Lower : Body            : Upper : Active
         1 :   0.0 :      x[1] - 5*z :   0.0 :   True
	 8 :   0.0 : x[2] - z * y[2] :   0.0 :   True
   


Pyomo Command
-------------

.. doctest::

    >>> print('Hello World')
    Hello World
