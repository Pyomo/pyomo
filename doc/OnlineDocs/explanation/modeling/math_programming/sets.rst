Sets
====

.. currentmodule:: pyomo.environ

Declaration
-----------

Sets can be declared using instances of the :class:`Set` and :class:`RangeSet`
classes or by
assigning set expressions.  The simplest set declaration creates a set
and postpones creation of its members:

.. doctest::
   :hide:

   >>> import pyomo.environ as pyo
   >>> model = pyo.ConcreteModel()

.. testcode::

   model.A = pyo.Set()

.. doctest::
   :hide:

   >>> # Add some data to the Set
   >>> model.A.update([1,2,3])

The :class:`Set` class takes optional arguments such as:

- ``dimen`` = Dimension of the members of the set
- ``doc`` = String describing the set
- ``filter`` = A Boolean function used during construction to indicate if a
  potential new member should be assigned to the set
- ``initialize`` = An iterable containing the initial members of the Set, or
  function that returns an iterable of the initial members the set.
- ``ordered`` = A Boolean indicator that the set is ordered; the default is ``True``
- ``validate`` = A Boolean function that validates new member data
- ``within`` = Set used for validation; it is a super-set of the set being declared.

In general, Pyomo attempts to infer the "dimensionality" of Set
components (that is, the number of apparent indices) when they are
constructed.  However, there are situations where Pyomo either cannot
detect a dimensionality (e.g., a :class:`Set` that was not initialized with any
members), or you the user may want to assert the dimensionality of the
set.  This can be accomplished through the ``dimen`` keyword.  For
example, to create a set whose members will be tuples with two items, one
could write:

.. testcode::

   model.B = pyo.Set(dimen=2)

To create a set of all the numbers in set ``model.A`` doubled, one could
use

.. testcode::

   def DoubleA_init(model):
       return (i*2 for i in model.A)
   model.C = pyo.Set(initialize=DoubleA_init)

As an aside we note that as always in Python, there are lot of ways to
accomplish the same thing. Also, note that this will generate an error
if ``model.A`` contains elements for which multiplication times two is
not defined.

The ``initialize`` option can accept any Python iterable, including a
``set``, ``list``, or ``tuple``.  This data may be returned from a
function or specified directly as in

.. testcode::

   model.D = pyo.Set(initialize=['red', 'green', 'blue'])

The ``initialize`` option can also specify either a generator or a
function to specify the Set members.  In the case of a generator, all
data yielded by the generator will become the initial set members:

.. testcode::

   def X_init(m):
       for i in range(10):
           yield 2*i+1
   model.X = pyo.Set(initialize=X_init)

For initialization functions, Pyomo supports two signatures.  In the
first, the function returns an iterable (``set``, ``list``, or
``tuple``) containing the data with which to initialize the Set:

.. testcode::

   def Y_init(m):
       return [2*i+1 for i in range(10)]
   model.Y = pyo.Set(initialize=Y_init)

In the second signature, the function is called for each element,
passing the element number in as an extra argument.  This is repeated
until the function returns the special value ``Set.End``:

.. testcode::

   def Z_init(model, i):
       if i > 10:
           return pyo.Set.End
       return 2*i+1
   model.Z = pyo.Set(initialize=Z_init)

Note that the element number starts with 1 and not 0:

.. doctest::

   >>> model.X.pprint()
   X : Size=1, Index=None, Ordered=Insertion
       Key  : Dimen : Domain : Size : Members
       None :     1 :    Any :   10 : {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}
   >>> model.Y.pprint()
   Y : Size=1, Index=None, Ordered=Insertion
       Key  : Dimen : Domain : Size : Members
       None :     1 :    Any :   10 : {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}
   >>> model.Z.pprint()
   Z : Size=1, Index=None, Ordered=Insertion
       Key  : Dimen : Domain : Size : Members
       None :     1 :    Any :   10 : {3, 5, 7, 9, 11, 13, 15, 17, 19, 21}

Additional information about iterators for set initialization is in the
[PyomoBookIII]_ book.

.. note::

   For Abstract models, data specified in an input file or through the
   ``data`` argument to :meth:`AbstractModel.create_instance()` will
   override the data
   specified by the initialize options.

If sets are given as arguments to :class:`Set` without keywords, they are
interpreted as indexes for an array of sets. For example, to create an
array of sets that is indexed by the members of the set ``model.A``, use:

.. testcode::

   model.E = pyo.Set(model.A)

Arguments can be combined. For example, to create an array of sets,
indexed by set ``model.A`` where each set contains three dimensional
members, use:

.. testcode::

   model.F = pyo.Set(model.A, dimen=3)

The ``initialize`` option can be used to create a set that contains a
sequence of numbers, but the :class:`RangeSet` class provides a concise
mechanism for simple sequences. This class takes as its arguments a
start value, a final value, and a step size. If the :class:`RangeSet` has
only a single argument, then that value defines the final value in the
sequence; the first value and step size default to one. If two values
given, they are the first and last value in the sequence and the step
size defaults to one. For example, the following declaration creates a
set with the numbers 1.5, 5 and 8.5:

.. testcode::

   model.G = pyo.RangeSet(1.5, 10, 3.5)

.. doctest::
   :hide:

   >>> model.G.pprint()
   G : Dimen=1, Size=3, Bounds=(1.5, 8.5)
       Key  : Finite : Members
       None :   True : [1.5], [5.0], [8.5]

Operations
-----------

Sets may also be created by storing the result of *set operations* using
other Pyomo sets.  Pyomo supports set operations including union, intersection,
difference, and symmetric difference:

.. testcode::

    model.I = model.A | model.D # union
    model.J = model.A & model.D # intersection
    model.K = model.A - model.D # difference
    model.L = model.A ^ model.D # exclusive-or

For example, the cross-product operator is the asterisk (*).  To define
a new set ``M`` that is the cross product of sets ``B`` and ``C``, one
could use

.. testcode::

   model.M = model.B * model.C

This creates a *virtual* set that holds references to the original sets,
so any updates to the original sets (``B`` and ``C``) will be reflected
in the new set (``M``).  In contrast, you can also create a *concrete*
set, which directly stores the values of the cross product at the time
of creation and will *not* reflect subsequent changes in the original
sets with:

.. testcode::

   model.M_concrete = pyo.Set(initialize=model.B * model.C)

Finally, you can indicate that the members of a set are restricted to be in the
cross product of two other sets, one can use the ``within`` keyword:

.. testcode::

   model.N = pyo.Set(within=model.B * model.C)

Predefined Virtual Sets
-----------------------

For use in specifying domains for sets, parameters and variables, Pyomo
provides the following pre-defined virtual sets:

- ``Any`` = all possible values
- ``Reals`` = floating point values
- ``PositiveReals`` = strictly positive floating point values
- ``NonPositiveReals`` = non-positive floating point values
- ``NegativeReals`` = strictly negative floating point values
- ``NonNegativeReals`` = non-negative floating point values
- ``PercentFraction`` = floating point values in the interval [0,1]
- ``UnitInterval`` = alias for PercentFraction
- ``Integers`` = integer values
- ``PositiveIntegers`` = positive integer values
- ``NonPositiveIntegers`` = non-positive integer values
- ``NegativeIntegers`` = negative integer values
- ``NonNegativeIntegers`` = non-negative integer values
- ``Boolean`` = Boolean values, which can be represented as False/True, 0/1,
  ’False’/’True’ and ’F’/’T’
- ``Binary`` = the integers {0, 1}

For example, if the set ``model.O`` is declared to be within the virtual
set ``NegativeIntegers`` then an attempt to add anything other than a
negative integer will result in an error. Here is the declaration:

.. testcode::

   model.O = pyo.Set(within=pyo.NegativeIntegers)

.. _Isinglecomm.py:

Sparse Index Sets
-----------------

Sets provide indexes for parameters, variables and other sets. Index set
issues are important for modelers in part because of efficiency
considerations, but primarily because the right choice of index sets can
result in very natural formulations that are conducive to understanding
and maintenance. Pyomo leverages Python to provide a rich collection of
options for index set creation and use.

The choice of how to represent indexes often depends on the application
and the nature of the instance data that are expected. To illustrate
some of the options and issues, we will consider problems involving
networks. In many network applications, it is useful to declare a set of
nodes, such as

.. testcode::

   model.Nodes = pyo.Set()

.. doctest::
   :hide:

   >>> model.Nodes.update(['a','b','c'])

and then a set of arcs can be created with reference to the nodes.

Consider the following simple version of minimum cost flow problem:

.. math::
   :nowrap:

   \[\begin{array}{lll}
     \mbox{minimize} & \sum_{a \in \mathcal{A}} c_{a}x_{a} \\
     \mbox{subject to:} & S_{n} + \sum_{(i,n) \in \mathcal{A}}x_{(i,n)} &  \\
			& -D_{n} - \sum_{(n,j) \in \mathcal{A}}x_{(n,j)} & n \in \mathcal{N} \\
			& x_{a} \geq 0, &  a \in \mathcal{A}
   \end{array}\]

where

- Set: Nodes :math:`\equiv \mathcal{N}`
- Set: Arcs :math:`\equiv \mathcal{A} \subseteq \mathcal{N} \times \mathcal{N}`
- Var: Flow on arc :math:`(i,j)` :math:`\equiv x_{i,j},\; (i,j) \in \mathcal{A}`
- Param: Flow Cost on arc :math:`(i,j)` :math:`\equiv c_{i,j},\; (i,j) \in \mathcal{A}`
- Param: Demand at node latexmath:`i` :math:`\equiv D_{i},\; i \in \mathcal{N}`
- Param: Supply at node latexmath:`i` :math:`\equiv S_{i},\; i \in \mathcal{N}`

In the simplest case, the arcs can just be the cross product of the
nodes, which is accomplished by the definition

.. testcode::

   model.Arcs = model.Nodes*model.Nodes

that creates a set with two dimensional members.  For applications where
all nodes are always connected to all other nodes this may
suffice. However, issues can arise when the network is not fully
dense. For example, the burden of avoiding flow on arcs that do not
exist falls on the data file where high-enough costs must be provided
for those arcs.  Such a scheme is not very elegant or robust.

For many network flow applications, it might be better to declare the
arcs using

.. doctest::
   :hide:

   >>> del model.Arcs

.. testcode::

   model.Arcs = pyo.Set(dimen=2)

or

.. doctest::
   :hide:

   >>> del model.Arcs

.. testcode::

   model.Arcs = pyo.Set(within=model.Nodes*model.Nodes)

where the difference is that the first version will provide error
checking as data is assigned to the set elements. This would enable
specification of a sparse network in a natural way. But this results in
a need to change the ``FlowBalance`` constraint because as it was
written in the simple example, it sums over the entire set of nodes for
each node. One way to remedy this is to sum only over the members of the
set ``model.arcs`` as in

.. testcode::

   def FlowBalance_rule(m, node):
       return m.Supply[node] \
           + sum(m.Flow[i, node] for i in m.Nodes if (i,node) in m.Arcs) \
           - m.Demand[node] \
           - sum(m.Flow[node, j] for j in m.Nodes if (j,node) in m.Arcs) \
           == 0

.. doctest::
   :hide:

   >>> model.Demand = pyo.Param(model.Nodes, default=1)
   >>> model.Supply = pyo.Param(model.Nodes, default=2)
   >>> model.Flow = pyo.Var(model.Nodes, model.Nodes)
   >>> model.Arcs.update([('a','b'),('b','c'),('c','a')])
   >>> model._flow_bal_1 = pyo.Constraint(model.Nodes, rule=FlowBalance_rule)

This will be OK unless the number of nodes becomes very large for a
sparse network, then the time to generate this constraint might become
an issue (admittely, only for very large networks, but such networks do
exist).

Another method, which comes in handy in many network applications, is to
have a set for each node that contain the nodes at the other end of arcs
going to the node at hand and another set giving the nodes on out-going
arcs. If these sets are called ``model.NodesIn`` and ``model.NodesOut``
respectively, then the flow balance rule can be re-written as

.. testcode::

   def FlowBalance_rule(m, node):
       return m.Supply[node] \
           + sum(m.Flow[i, node] for i in m.NodesIn[node]) \
           - m.Demand[node] \
           - sum(m.Flow[node, j] for j in m.NodesOut[node]) \
           == 0

The data for ``NodesIn`` and ``NodesOut`` could be added to the input
file, and this may be the most efficient option.

For all but the largest networks, rather than reading ``Arcs``,
``NodesIn`` and ``NodesOut`` from a data file, it might be more elegant
to read only ``Arcs`` from a data file and declare ``model.NodesIn``
with an ``initialize`` option specifying the creation as follows:

.. testcode::

    def NodesIn_init(m, node):
        for i, j in m.Arcs:
            if j == node:
                yield i
    model.NodesIn = pyo.Set(model.Nodes, initialize=NodesIn_init)

.. doctest::
   :hide:

    >>> model.NodesOut = pyo.Set(model.Nodes)
    >>> model._flow_bal_2 = pyo.Constraint(model.Nodes, rule=FlowBalance_rule)

with a similar definition for ``model.NodesOut``.  This code creates a
list of sets for ``NodesIn``, one set of nodes for each node. The full
model is:

.. testcode::

    import pyomo.environ as pyo

    model = pyo.AbstractModel()

    model.Nodes = pyo.Set()
    model.Arcs = pyo.Set(dimen=2)

    def NodesOut_init(m, node):
        for i, j in m.Arcs:
            if i == node:
                yield j
    model.NodesOut = pyo.Set(model.Nodes, initialize=NodesOut_init)

    def NodesIn_init(m, node):
        for i, j in m.Arcs:
            if j == node:
                yield i
    model.NodesIn = pyo.Set(model.Nodes, initialize=NodesIn_init)

    model.Flow = pyo.Var(model.Arcs, domain=pyo.NonNegativeReals)
    model.FlowCost = pyo.Param(model.Arcs)

    model.Demand = pyo.Param(model.Nodes)
    model.Supply = pyo.Param(model.Nodes)

    def Obj_rule(m):
        return pyo.summation(m.FlowCost, m.Flow)
    model.Obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    def FlowBalance_rule(m, node):
        return m.Supply[node] \
            + sum(m.Flow[i, node] for i in m.NodesIn[node]) \
            - m.Demand[node] \
            - sum(m.Flow[node, j] for j in m.NodesOut[node]) \
            == 0
    model.FlowBalance = pyo.Constraint(model.Nodes, rule=FlowBalance_rule)

for this model, a toy data file (in AMPL "``.dat``" format) would be:

.. literalinclude:: /src/scripting/Isinglecomm.dat
   :language: text

.. doctest::
   :hide:

   >>> inst = model.create_instance('src/scripting/Isinglecomm.dat')

This can also be done somewhat more efficiently, and perhaps more clearly,
using a :class:`BuildAction` (for more information, see :ref:`BuildAction`):

.. doctest::
   :hide:

   >>> model = inst
   >>> del model.NodesIn
   >>> del model.NodesOut

.. testcode::

   model.NodesOut = pyo.Set(model.Nodes, within=model.Nodes)
   model.NodesIn = pyo.Set(model.Nodes, within=model.Nodes)

   def Populate_In_and_Out(model):
       # loop over the arcs and record the end points
       for i, j in model.Arcs:
           model.NodesIn[j].add(i)
           model.NodesOut[i].add(j)

   model.In_n_Out = pyo.BuildAction(rule=Populate_In_and_Out)

Sparse Index Sets Example
^^^^^^^^^^^^^^^^^^^^^^^^^

One may want to have a constraint that holds

.. math::
   \forall \; i \in I, k \in K, v \in V_k

There are many ways to accomplish this, but one good way is to create a
set of tuples composed of all ``model.k, model.V[k]`` pairs.  This
can be done as follows:

.. testcode::
   :hide:

   model = pyo.ConcreteModel()

   model.I = pyo.RangeSet(3)
   model.K = pyo.RangeSet(5)

   def V_init(m, k):
       return range(0, k, 2)
   model.V = pyo.Set(model.K, initialize=V_init)

.. testcode::

   def kv_init(m):
       return ((k,v) for k in m.K for v in m.V[k])
   model.KV = pyo.Set(dimen=2, initialize=kv_init)

We can now create the constraint :math:`x_{i,k,v} \leq a_{i,k}y_i
\;\forall\; i \in I, k \in K, v \in V_k` with:

.. testcode::

   model.a = pyo.Param(model.I, model.K, default=1)

   model.y = pyo.Var(model.I)
   model.x = pyo.Var(model.I, model.KV)

   def c1_rule(m, i, k, v):
       return m.x[i,k,v] <= m.a[i,k]*m.y[i]
   model.c1 = pyo.Constraint(model.I, model.KV, rule=c1_rule)
