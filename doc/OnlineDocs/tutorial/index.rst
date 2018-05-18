Advanced Tutorial
=================

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
:class:`Param<pyomo.core.base.param.Param>`,
:class:`Var<pyomo.core.base.var.Var>`,
:class:`Constraint<pyomo.core.base.constraint.Constraint>`,
:class:`Objective<pyomo.core.base.objective.Objective>`,
:class:`Set<pyomo.core.base.sets.Set>`,
and
:class:`Block<pyomo.core.base.block.Block>`. Each of these may be indexed.

.. doctest::

   >>> import pyomo.environ as pe
   >>> m = pe.ConcreteModel()
   >>> m.a = pe.Set(initialize=[1, 2, 3])
   >>> m.x = pe.Var(m.a, initialize=0, bounds=(-10,10))
   >>> m.y = pe.Var(m.a)
   >>> def c_rule(m, i):
   ...     return m.x[i] >= m.y[i]
   >>> m.c = pe.Constraint(m.a, rule=c_rule)
   >>> m.c.pprint()
   c : Size=3, Index=a, Active=True
       Key : Lower : Body        : Upper : Active
         1 :  -Inf : y[1] - x[1] :   0.0 :   True
         2 :  -Inf : y[2] - x[2] :   0.0 :   True
         3 :  -Inf : y[3] - x[3] :   0.0 :   True

The index specifies the set of *allowable members* of the component.  In
the case of :class:`Var<pyomo.core.base.var.Var>`, the constructor will
automatically create variables for each member of the index.  Other
components (like
:class:`Constraint<pyomo.core.base.constraint.Constraint`>) leverage a
*rule*, which is called by the constructor for every member of the
index, the return value of which dictates whether or not to create
the corresponding modeling object.  Beyond facilitating the construction
of large structured models, discrete indexing sets provide error
checking, ensuring that requested modeling objects are allowed by the
index:

.. doctest::

   >>> m.z = pe.Var()
   >>> m.c[4] = m.x[1] == 5 * m.z
   Traceback (most recent call last):
     ...
   KeyError: "Index '4' is not valid for indexed component 'c'"

This helps prevent many common mistakes.  To add new objects to a
component, the new index must first be added to the underlying index
set:

.. doctest::

   >>> m.a.add(4)
   >>> m.c[4] = m.x[1] == 5 * m.z

However, it is sometimes useful to allow a more flexible form of
indexing using non-iterable sets.  For example, an indexed component may
be made to behave like a dictionary by indexing it using the `Any` set.
This set admits any hashable object as a member.

.. doctest::

   >>> m.c2 = pe.Constraint(pe.Any)
   >>> m.c2[1] = m.x[1] == 5 * m.z
   >>> m.c2[8] = m.x[2] == m.z * m.y[2]
   >>> m.c2.pprint()
   c2 : Size=2, Index=Any, Active=True
       Key : Lower : Body          : Upper : Active
         1 :   0.0 :    x[1] - 5*z :   0.0 :   True
         8 :   0.0 : x[2] - z*y[2] :   0.0 :   True

.. note::

   It it important that the component construction not iterate over the
   non-iterable set.  For most components, simply omitting the `rule=`
   argument is sufficient.  :class:`Var<pyomo.core.base.var.Var>`
   requires the `dense=False` argument so that the constructor does not
   iterate over the non-iterable set.

.. doctest::

   >>> m.v = pe.Var(pe.Any, dense=False)
   >>> m.c2[2] = m.v[1] + m.v[2] == 0
   >>> m.v.pprint()
   v : Size=2, Index=Any
       Key : Lower : Value : Upper : Fixed : Stale : Domain
         1 :  None :  None :  None : False :  True :  Reals
         2 :  None :  None :  None : False :  True :  Reals
   >>> m.c2.pprint()
   c2 : Size=3, Index=Any, Active=True
       Key : Lower : Body          : Upper : Active
         1 :   0.0 :    x[1] - 5*z :   0.0 :   True
         2 :   0.0 :   v[1] + v[2] :   0.0 :   True
         8 :   0.0 : x[2] - z*y[2] :   0.0 :   True

The following illustrates how to use `Any` with Blocks.

.. doctest::

   >>> m.b = pe.Block(pe.Any)
   >>> m.b['foo1'].x = pe.Var()
   >>> m.b['foo1'].y = pe.Var()
   >>> m.b['foo1'].c = pe.Constraint(expr=m.b['foo1'].x >= 5*m.b['foo1'].y)
   >>> m.b[1].x = pe.Var()


