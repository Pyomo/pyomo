Sets
====

Declaration
-----------

Sets can be declared using the `Set` and `RangeSet` functions or by
assigning set expressions.  The simplest set declaration creates a set
and postpones creation of its members:

.. literalinclude:: ../script_spy_files/Spy4Sets_Declare_set.spy
   :language: python

The ``Set`` function takes optional arguments such as:

- doc = String describing the set
- dimen = Dimension of the members of the set
- filter = A boolean function used during construction to indicate if a
  potential new member should be assigned to the set
- initialize = A function that returns the members to initialize the set.
- ordered = A boolean indicator that the set is ordered; the default is ``False``
- validate = A boolean function that validates new member data
- virtual = A boolean indicator that the set will never have elements;
  it is unusual for a modeler to create a virtual set; they are
  typically used as domains for sets, parameters and variables
- within = Set used for validation; it is a super-set of the set being declared.

One way to create a set whose members will be two dimensional is to use
the ``dimen`` argument:

.. literalinclude:: ../script_spy_files/Spy4Sets_Set_dim.spy
   :language: python

To create a set of all the numbers in set ``model.A`` doubled, one could
use

.. literalinclude:: ../script_spy_files/doubleA.py
   :language: python

As an aside we note that as always in Python, there are lot of ways to
accomplish the same thing. Also, note that this will generate an error
if ``model.A`` contains elements for which multiplication times two is
not defined.

The ``initialize`` option can refer to a Python set, which can be
returned by a function or given directly as in

.. literalinclude:: ../script_spy_files/Spy4Sets_Initialize_python_sets.spy
   :language: python

The ``initialize`` option can also specify a function that is applied
sequentially to generate set members. Consider the case of a simple
set. In this case, the initialization function accepts a set element
number and model and returns the set element associated with that
number:

.. literalinclude:: ../script_spy_files/Z_init.py
   :language: python

The ``Set.End`` return value terminates input to the set. Additional
information about iterators for set initialization is in the
[PyomoBookII]_ book.

.. note::

   Data specified in an input file will override the data specified by
   the initialize options.

If sets are given as arguments to ``Set`` without keywords, they are
interpreted as indexes for an array of sets. For example, to create an
array of sets that is indexed by the members of the set ``model.A``, use

.. literalinclude:: ../script_spy_files/Spy4Sets_Sets_without_keywords.spy
   :language: python

Arguments can be combined. For example, to create an array of sets with
three dimensional members indexed by set ``model.A``, use

.. literalinclude:: ../script_spy_files/Spy4Sets_Combined_arguments.spy
   :language: python

The ``initialize`` option can be used to create a set that contains a
sequence of numbers, but the ``RangeSet`` function provides a concise
mechanism for simple sequences. This function takes as its arguments a
start value, a final value, and a step size. If the ``RangeSet`` has
only a single argument, then that value defines the final value in the
sequence; the first value and step size default to one. If two values
given, they are the first and last value in the sequence and the step
size defaults to one. For example, the following declaration creates a
set with the numbers 1.5, 5 and 8.5:

.. literalinclude:: ../script_spy_files/Spy4Sets_RangeSet_simple_sequence.spy
   :language: python

Operations
-----------

Sets may also be created by assigning other Pyomo sets as in these
examples that also illustrate the set operators union, intersection,
difference, and exclusive-or:

.. literalinclude:: ../script_spy_files/Spy4Sets_Set_operators.spy
   :language: python

The cross-product operator is the asterisk (*). For example, to assign a
set the cross product of two other sets, one could use

.. literalinclude:: ../script_spy_files/Spy4Sets_Set_cross_product.spy
   :language: python

or to indicate the the members of a set are restricted to be in the
cross product of two other sets, one could use

.. literalinclude:: ../script_spy_files/Spy4Sets_Restrict_to_crossproduct.spy
   :language: python

The cross-product operator is the asterisk (*).  For example, to create
a set that contains the cross-product of sets A and B, use

.. literalinclude:: ../script_spy_files/Spy4Sets_Assign_to_crossproduct.spy
   :language: python

to instead create a set that can contain a subset of the members of this
cross-product, use

.. literalinclude:: ../script_spy_files/Spy4Sets_Contain_crossproduct_subset.spy
   :language: python

Predefined Virtual Sets
-----------------------

For use in specifying domains for sets, parameters and variables, Pyomo
provides the following pre-defined virtual sets:

- Any = all possible values
- Reals = floating point values
- PositiveReals = strictly positive floating point values
- NonPositiveReals = non-positive floating point values
- NegativeReals = strictly negative floating point values
- NonNegativeReals = non-negative floating point values
- PercentFraction = floating point values in the interval [0,1]
- UnitInterval = alias for PercentFraction
- Integers = integer values
- PositiveIntegers = positive integer values
- NonPositiveIntegers = non-positive integer values
- NegativeIntegers = negative integer values
- NonNegativeIntegers = non-negative integer values
- Boolean = boolean values, which can be represented as False/True, 0/1,
  ’False’/’True’ and ’F’/’T’
- Binary = same as boolean

For example, if the set ``model.M`` is declared to be within the virtual
set ``NegativeIntegers`` then an attempt to add anything other than a
negative integer will result in an error. Here is the declaration:

.. literalinclude:: ../script_spy_files/Spy4Sets_Predefined_set_example.spy
   :language: python

.. _Isinglecomm.py:

Sparse Index Sets
-----------------

Sets provide indexes for parameters, variables and other sets. Index set
issues are important for modelers in part because of efficiency
considerations, but primarily because the right choice of index sets can
result in very natural formulations that are condusive to understanding
and maintenance. Pyomo leverages Python to provide a rich collection of
options for index set creation and use.

The choice of how to represent indexes often depends on the application
and the nature of the instance data that are expected. To illustrate
some of the options and issues, we will consider problems involving
networks. In many network applications, it is useful to declare a set of
nodes, such as

.. literalinclude:: ../script_spy_files/Spy4Sets_Declare_nodes.spy
   :language: python

and then a set of arcs can be created with reference to the nodes.

Consider the following simple version of minimum cost flow problem:

.. math::
   :nowrap:

	\begin{array}{lll}
	\mbox{minimize} & \sum_{a \in \mathcal{A}} c_{a}x_{a} \\
	\mbox{subject to:} & S_{n} + \sum_{(i,n) \in \mathcal{A}}x_{(i,n)} &  \\
					& -D_{n} - \sum_{(n,j) \in \mathcal{A}}x_{(n,j)} & n \in \mathcal{N} \\
					& x_{a} \geq 0, &  a \in \mathcal{A}
	\end{array}

where

- Set: Nodes :math:`\equiv \mathcal{N}`
- Set: Arcs :math:`\equiv \mathcal{A} \subseteq \mathcal{N} \times \mathcal{N}`
- Var: Flow on arc :math:`(i,j)` :math:`\equiv x_{i,j},\; (i,j) \in \mathcal{A}`
- Param: Flow Cost on arc :math:`(i,j)` :math:`\equiv c_{i,j},\; (i,j) \in \mathcal{A}`
- Param: Demand at node latexmath:`i` :math:`\equiv D_{i},\; i \in \mathcal{N}`
- Param: Supply at node latexmath:`i` :math:`\equiv S_{i},\; i \in \mathcal{N}`

In the simplest case, the arcs can just be the cross product of the
nodes, which is accomplished by the definition

.. literalinclude:: ../script_spy_files/Spy4Sets_Declare_arcs_crossproduct.spy
   :language: python

that creates a set with two dimensional members.  For applications where
all nodes are always connected to all other nodes this may
suffice. However, issues can arise when the network is not fully
dense. For example, the burden of avoiding flow on arcs that do not
exist falls on the data file where high-enough costs must be provided
for those arcs.  Such a scheme is not very elegant or robust.

For many network flow applications, it might be better to declare the
arcs using

.. literalinclude:: ../script_spy_files/Spy4Sets_Declare_arcs_within.spy
   :language: python

or

.. literalinclude:: ../script_spy_files/Spy4Sets_Declare_arcs_dimen.spy
   :language: python

where the difference is that the first version will provide error
checking as data is assigned to the set elements. This would enable
specification of a sparse network in a natural way. But this results in
a need to change the ``FlowBalance`` constraint because as it was
written in the simple example, it sums over the entire set of nodes for
each node. One way to remedy this is to sum only over the members of the
set ``model.arcs`` as in

.. literalinclude:: ../script_spy_files/FlowBalance_rule.py
   :language: python

This will be OK unless the number of nodes becomes very large for a
sparse network, then the time to generate this constraint might become
an issue (admittely, only for very large networks, but such networks do
exist).

Another method, which comes in handy in many network applications, is to
have a set for each node that contain the nodes at the other end of arcs
going to the node at hand and another set giving the nodes on out-going
arcs. If these sets are called ``model.NodesIn`` and ``model.NodesOut``
respectively, then the flow balance rule can be re-written as

.. literalinclude:: ../script_spy_files/FlowBalance_rule2.py
   :language: python

The data for ``NodesIn`` and ``NodesOut`` could be added to the input
file, and this may be the most efficient option.

For all but the largest networks, rather than reading ``Arcs``,
``NodesIn`` and ``NodesOut`` from a data file, it might be more elegant
to read only ``Arcs`` from a data file and declare ``model.NodesIn``
with an ``initialize`` option specifying the creation as follows:

.. literalinclude:: ../script_spy_files/NodesIn_init.py
   :language: python

with a similar definition for ``model.NodesOut``.  This code creates a
list of sets for ``NodesIn``, one set of nodes for each node. The full
model is:

.. literalinclude:: ../script_spy_files/Isinglecomm.py
   :language: python

for this model, a toy data file would be:

.. literalinclude:: ../script_spy_files/Isinglecomm.dat
   :language: python

This can be done somewhat more efficiently, and perhaps more clearly,
using a BuildAction as shown in Isinglebuild.py in :ref:Isinglebuild.py.

Sparse Index Sets Example
^^^^^^^^^^^^^^^^^^^^^^^^^

One may want to have a constraint that holds

>>> for i in model.I, k in model.K, v in model.V[k] # doctest: +SKIP

There are many ways to accomplish this, but one good way is to create a
set of tuples composed of all of ``model.k, model.V[k]`` pairs.  This
can be done as follows:

.. literalinclude:: ../script_spy_files/Spy4Sets_Define_constraint_tuples.spy
   :language: python

So then if there was a constraint defining rule such as

>>> def MyC_rule(model, i, k, v): # doctest: +SKIP
>>>    return ...                 # doctest: +SKIP

Then a constraint could be declared using

.. literalinclude:: ../script_spy_files/Spy4Sets_Define_another_constraint.spy
   :language: python

Here is the first few lines of a model that illustrates this:

.. literalinclude:: ../script_spy_files/Spy4Sets_Declare_constraints_example.spy
   :language: python
