Pyomo Network
=============

Pyomo Network is a package that allows users to easily represent their model
as a connected network of units. Units are blocks that contain ports, which
contain variables, that are connected to other ports via arcs. The connection
of two ports to each other via an arc typically represents a set of constraints
equating each member of each port to each other, however there exist other
connection rules as well, in addition to support for custom rules. Pyomo
Network also includes a model transformation that will automatically expand
the arcs and generate the appropriate constraints to produce an algebraic
model that a solver can handle. Furthermore, the package also introduces a
generic sequential decomposition tool that can leverage the modeling
components to decompose a model and compute each unit in the model in a
logically ordered sequence.

Modeling Components
-------------------

Pyomo Network introduces two new modeling components to Pyomo:

.. autosummary::
    :nosignatures:

    pyomo.network.Port
    pyomo.network.Arc

Port
****

.. autoclass:: pyomo.network.Port
    :members:
    :exclude-members: construct, display

.. autoclass:: pyomo.network.port._PortData
    :members:
    :special-members: __getattr__
    :exclude-members: set_value

The following code snippet shows examples of declaring and using a
:py:class:`Port <pyomo.network.Port>` component on a
concrete Pyomo model:

.. doctest::

   >>> from pyomo.environ import *
   >>> from pyomo.network import *
   >>> m = ConcreteModel()
   >>> m.x = Var()
   >>> m.y = Var(['a', 'b']) # can be indexed
   >>> m.z = Var()
   >>> m.e = 5 * m.z # you can add Pyomo expressions too
   >>> m.w = Var()

   >>> m.p = Port()
   >>> m.add(m.x) # implicitly name the port member "x"
   >>> m.add(m.y, "foo") # name the member "foo"
   >>> m.add(m.e, rule=Port.Extensive) # specify a rule
   >>> m.add(m.w, rule=Port.Extensive, write_var_sum=False) # keyword arg

Arc
***

.. autoclass:: pyomo.network.Arc
    :members:
    :exclude-members: construct

.. autoclass:: pyomo.network.arc._ArcData
    :members:
    :special-members: __getattr__

The following code snippet shows examples of declaring and using an
:py:class:`Arc <pyomo.network.Arc>` component on a
concrete Pyomo model:

.. doctest::

   >>> from pyomo.environ import *
   >>> from pyomo.network import *
   >>> m = ConcreteModel()
   >>> m.x = Var()
   >>> m.y = Var(['a', 'b'])
   >>> m.u = Var()
   >>> m.v = Var(['a', 'b']) # indexes need to match

   >>> m.p = Port(initialize=[m.x, m.y])
   >>> m.q = Port(initialize={"x": m.u, "y": m.v}) # names need to match
   >>> m.a = Arc(source=m.p, destination=m.q) # directed
   >>> m.b = Arc(ports=(m.p, m.q)) # undirected
   >>> m.c = Arc(ports=(m.p, m.q), directed=True) # directed
