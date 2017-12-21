Sets
====

Declaration
-----------

Sets can be declared using the `Set` and `RangeSet` functions or by
assigning set expressions.  The simplest set declaration creates
a set and postpones creation of its members:

>>> model.A = Set()

The ``Set`` function takes optional arguments such as:

.. autosummary::
   :nosignatures:
   
- doc = String describing the set
- dimen = Dimension of the members of the set
- filter = A boolean function used during construction to indicate if a potential new member should be assigned to the set
- initialize = A function that returns the members to initialize the set.
- ordered = A boolean indicator that the set is ordered; the default is ``False``
- validate = A boolean function that validates new member data
- virtual = A boolean indicator that the set will never have elements; it is unusual for a modeler to create a virtual set; they are typically used as domains for sets, parameters and variables
- within = Set used for validation; it is a super-set of the set being declared.

One way to create a set whose members will be two dimensional is to use
the ``dimen`` argument:

>>> model.B = Set(dimen=2)

To create a set of all the numbers in set ``model.A`` doubled, one could use




