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
    >>> m.p.add(m.x) # implicitly name the port member "x"
    >>> m.p.add(m.y, "foo") # name the member "foo"
    >>> m.p.add(m.e, rule=Port.Extensive) # specify a rule
    >>> m.p.add(m.w, rule=Port.Extensive, write_var_sum=False) # keyword arg

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
    >>> m.v = Var(['a', 'b'])
    >>> m.w = Var()
    >>> m.z = Var(['a', 'b']) # indexes need to match

    >>> m.p = Port(initialize=[m.x, m.y])
    >>> m.q = Port(initialize={"x": m.u, "y": m.v})
    >>> m.r = Port(initialize={"x": m.w, "y": m.z}) # names need to match
    >>> m.a = Arc(source=m.p, destination=m.q) # directed
    >>> m.b = Arc(ports=(m.p, m.q)) # undirected
    >>> m.c = Arc(ports=(m.p, m.q), directed=True) # directed
    >>> m.d = Arc(src=m.p, dest=m.q) # aliases work
    >>> m.e = Arc(source=m.r, dest=m.p) # ports can have both in and out

Arc Expansion Transformation
----------------------------

The examples above show how to declare and instantiate a
:py:class:`Port <pyomo.network.Port>` and an
:py:class:`Arc <pyomo.network.Arc>`. These two components form the basis of
the higher level representation of a connected network with sets of related
variable quantities. Once a network model has been constructed, Pyomo Network
implements a transformation that will expand all (active) arcs on the model
and automatically generate the appropriate constraints. The constraints
created for each port member will be indexed by the same indexing set as
the port member itself.

During transformation, a new block is created on the model for each arc
(located on the arc's parent block), which serves to contain all of the
auto generated constraints for that arc. At the end of the
transformation, a reference is created on the arc that points to this
new block, available via the arc property `arc.expanded_block`.

The constraints produced by this transformation depend on the rule assigned
for each port member and can be different between members on the same port.
For example, you can have two different members on a port where one member's
rule is :py:func:`Port.Equality <pyomo.network.Port.Equality>` and the other
member's rule is :py:func:`Port.Extensive <pyomo.network.Port.Extensive>`.

:py:func:`Port.Equality <pyomo.network.Port.Equality>` is the default rule
for port members. This rule simply generates equality constraints on the
expanded block between the source port's member and the destination port's
member. Another implemented expansion method is
:py:func:`Port.Extensive <pyomo.network.Port.Extensive>`, which essentially
represents implied splitting and mixing of certain variable quantities.
Users can refer to the documentation of the static method itself for more
details on how this implicit splitting and mixing is implemented.
Additionally, should users desire, the expansion API supports custom rules
that can be implemented to generate whatever is needed for special cases.

The following code demonstrates how to call the transformation to expand
the arcs on a model:

.. doctest::

    >>> from pyomo.environ import *
    >>> from pyomo.network import *
    >>> m = ConcreteModel()
    >>> m.x = Var()
    >>> m.y = Var(['a', 'b'])
    >>> m.u = Var()
    >>> m.v = Var(['a', 'b'])

    >>> m.p = Port(initialize=[m.x, (m.y, Port.Extensive)]) # rules must match
    >>> m.q = Port(initialize={"x": m.u, "y": (m.v, Port.Extensive)})
    >>> m.a = Arc(source=m.p, destination=m.q)

    >>> TransformationFactory("network.expand_arcs").apply_to(m)

Sequential Decomposition
------------------------

Pyomo Network implements a generic
:py:class:`SequentialDecomposition <pyomo.network.SequentialDecomposition>`
tool that can be used to compute each unit in a network model in a logically
ordered sequence.

The sequential decomposition procedure is commenced via the
:py:func:`run <pyomo.network.SequentialDecomposition.run>` method.

Creating a Graph
****************

To begin this procedure, the Pyomo Network model is first utilized to create
a networkx `MultiDiGraph` by adding edges to the graph for every arc on the
model, where the nodes of the graph are the parent blocks of the source and
destination ports. This is done via the
:py:func:`create_graph <pyomo.network.SequentialDecomposition.create_graph>`
method, which requires all arcs on the model to be both directed and already
expanded. The `MultiDiGraph` class of networkx supports both direccted edges
as well as having multiple edges between the same two nodes, so users can
feel free to connect as many ports as desired between the same two units.

Computation Order
*****************

The order of computation is then determined by treating the resulting graph
as a tree, starting at the roots of the tree, and making sure by the time
each node is reached, all of its predecessors have already been computed.
This is implemented through the :py:func:`calculation_order
<pyomo.network.SequentialDecomposition.calculation_order>` and
:py:func:`tree_order <pyomo.network.SequentialDecomposition.tree_order>`
methods. Before this, however, the procedure will first select a set of tear
edges, if necessary, such that every loop in the graph is torn, while
minimizing both the number of times any single loop is torn as well as the
total number of tears.

Tear Selection
**************

A set of tear edges can be selected in one of two ways. By default, a Pyomo
MIP model is created and optimized resulting in an optimal set of tear edges.
The implementation of this MIP model is based on a set of binary "torn"
variables for every edge in the graph, and constraints on every loop in the
graph that dictate that there must be at least one tear on the loop. Then
there are two objectives (represented by a doubly weighted objective). The
primary objective is to minimize the number of times any single loop is torn,
and then secondary to that is to minimize the total number of tears. This
process is implemented in the :py:func:`select_tear_mip
<pyomo.network.SequentialDecomposition.select_tear_mip>` method, which uses
the model returned from the :py:func:`select_tear_mip_model
<pyomo.network.SequentialDecomposition.select_tear_mip_model>` method.

Alternatively, there is the :py:func:`select_tear_heuristic
<pyomo.network.SequentialDecomposition.select_tear_heuristic>` method. This
uses a heuristic procedure that walks back and forth on the graph to find
every optimal tear set, and returns each equally optimal tear set it finds.
This method is much slower than the MIP method on larger models, but it
maintains some use in the fact that it returns every possible optimal tear set.

A custom tear set can be assigned before calling the
:py:func:`run <pyomo.network.SequentialDecomposition.run>` method. This is
useful so users can know what their tear set will be and thus what arcs will
require guesses for uninitialized values. See the
:py:func:`set_tear_set <pyomo.network.SequentialDecomposition.set_tear_set>`
method for details.

Running the Sequential Decomposition Procedure
**********************************************

After all of this computational order preparation, the sequential
decomposition procedure will then run through the graph in the order it
has determined. Thus, the `function` that was passed to the
:py:func:`run <pyomo.network.SequentialDecomposition.run>` method will be
called on every unit in sequence. This function can perform any arbitrary
operations the user desires. The only thing that
:py:class:`SequentialDecomposition <pyomo.network.SequentialDecomposition>`
expects from the function is that after returning from it, every variable
on every outgoing port of the unit will be specified (i.e. it will have a
set current value). Furthermore, the procedure guarantees to the user that
for every unit, before the function is called, every variable on every
incoming port of the unit will be fixed.

In between computing each of these units, port member values are passed
across existing arcs involving the unit currently being computed. This means
that after computing a unit, the expanded constraints from each arc coming
out of this unit will be satisfied, and the values on the respective
destination ports will be fixed at these new values. While running the
computational order, values are not passed across tear edges, as tear edges
represent locations in loops to stop computations (during iterations). This
process continues until all units in the network have been computed. This
concludes the "first pass run" of the network.

Guesses and Fixing Variables
****************************

When passing values across arcs while running the computational order,
values at the destinations of each of these arcs will be fixed at the
appropriate values. This is important to the fact that the procedure
guarantees every inlet variable will be fixed before calling the function.
However, since values are not passed across torn arcs, there is a need for
user-supplied guesses for those values. See the :py:func:`set_guesses_for
<pyomo.network.SequentialDecomposition.set_guesses_for>` method for details
on how to supply these values.

In addition to passing dictionaries of guesses for certain ports, users can
also assign current values to the variables themselves and the procedure
will pick these up and fix the variables in place. Alternatively, users can
utilize the `default_guess` option to specify a value to use as a default
guess for all free variables if they have no guess or current value. If a
free variable has no guess or current value and there is no default guess
option, then an error will be raised.

Similarly, if the procedure attempts to pass a value to a destination port
member but that port member is already fixed and its fixed value is different
from what is trying to be passed to it (by a tolerance specified by the
`almost_equal_tol` option), then an error will be raised. Lastly, if there
is more than one free variable in a constraint while trying to pass values
across an arc, an error will be raised asking the user to fix more variables
by the time values are passed across said arc.

Tear Convergence
****************

After completing the first pass run of the network, the sequential
decomposition procedure will proceed to converge all tear edges in the
network (unless the user specifies not to, or if there are no tears).
This process occurs separately for every strongly connected component (SCC)
in the graph, and the SCCs are computed in a logical order such that each
SCC is computed before other SCCs downstream of it (much like
:py:func:`tree_order <pyomo.network.SequentialDecomposition.tree_order>`).

There are two implemented methods for converging tear edges: direct
substitution and Wegstein acceleration. Both of these will iteratively run
the computation order until every value in every tear arc has converged to
within the specified tolerance. See the
:py:class:`SequentialDecomposition <pyomo.network.SequentialDecomposition>`
parameter documentation for details on what can be controlled about this
procedure.

The following code demonstrates basic usage of the
:py:class:`SequentialDecomposition <pyomo.network.SequentialDecomposition>`
class:

.. doctest::
    :skipif: not __import__("pyomo.network").network.decomposition.imports_available

    >>> from pyomo.environ import *
    >>> from pyomo.network import *
    >>> m = ConcreteModel()
    >>> m.unit1 = Block()
    >>> m.unit1.x = Var()
    >>> m.unit1.y = Var(['a', 'b'])
    >>> m.unit2 = Block()
    >>> m.unit2.x = Var()
    >>> m.unit2.y = Var(['a', 'b'])
    >>> m.unit1.port = Port(initialize=[m.unit1.x, (m.unit1.y, Port.Extensive)])
    >>> m.unit2.port = Port(initialize=[m.unit2.x, (m.unit2.y, Port.Extensive)])
    >>> m.a = Arc(source=m.unit1.port, destination=m.unit2.port)
    >>> TransformationFactory("network.expand_arcs").apply_to(m)

    >>> m.unit1.x.fix(10)
    >>> m.unit1.y['a'].fix(15)
    >>> m.unit1.y['b'].fix(20)

    >>> seq = SequentialDecomposition(tol=1.0E-3) # options can go to init
    >>> seq.options.select_tear_method = "heuristic" # or set them like so
    >>> # seq.set_tear_set([...]) # assign a custom tear set
    >>> # seq.set_guesses_for(m.unit.inlet, {...}) # choose guesses
    >>> def initialize(b):
    ...     # b.initialize()
    ...     pass
    ...
    >>> seq.run(m, initialize)

.. autoclass:: pyomo.network.SequentialDecomposition
    :members: set_guesses_for, set_tear_set, tear_set_arcs, indexes_to_arcs,
        run, create_graph, select_tear_mip, select_tear_mip_model,
        select_tear_heuristic, calculation_order, tree_order
