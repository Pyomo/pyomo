Constants
=========

Overview
--------
Constants are a key mechanism used to capture the relationship between variables.  In engineering design, constants are often defined by physics or operational limits.

The Constant constructor is a very thin wrapper on Pyomo :py:class:`Param <pyomo.environ.Param>`, and so experienced Pyomo users will not see any significant differences from base Pyomo.  


Construction
------------

Constants are constructed by 1) creating an instance of a new constant in a EDI Formulation and 2) passing out this newly constructed constant to be used in objective and constraint construction.  

.. py:function:: f.Constant(name, value, units, description='', size=None, within=None)

    Declares a constant in a pyomo.contrib.edi.formulation

   :param name: The name of the constant for the purposes of tracking in the formulation.  Commonly, this will be the same as the constant name in local namespace.
   :type  name: str
   :param value: The value of the constant.  For scalar constants, this should be a valid float or int for the specified domain.  For vector constants, this will most often also be a single float or int, but a dictionary of index-value pairs is also accepted as in accordance with base Pyomo.  NumPy arrays will be supported in a future release (see `this issue <https://github.com/codykarcher/pyomo/issues/1>`__)
   :type  value: float or int or dict
   :param units: The units of the constant.  Every entry in a vector constant must have the same units.  Entries of '', ' ', '-', 'None', and 'dimensionless' all become units.dimensionless
   :type  units: str or pyomo.core.base.units_container._PyomoUnit
   :param description: A description of the constant
   :type  description: str
   :param size: The size (or shape) of the constant.  Entries of 0, 1, and None all correspond to scalar constants.  Other integers correspond to vector constants.  Matrix and tensor constants are declared using lists of ints, ex: [10,10].  Matrix and tensor constants with a dimension of 1 (i.e., [10,10,1]) will be rejected as the extra dimension holds no meaningful value.  
   :type  size: int or list
   :param within: The domain of the constant (ex: Reals, Integers, etc).  Default of None constructs a constant in Reals.  This option should rarely be used.
   :type  within: pyomo set

   :return: The constant that was declared in the formulation
   :rtype: pyomo.core.base.param.ScalarParam or pyomo.core.base.param.IndexedParam


Relation to Pyomo Param
-----------------------

The arguments ``name``, ``within``, and ``bounds`` are directly passed to the Pyomo ``Param`` constructor, with some minor checking.  The ``value`` field is passed to the ``Param`` ``initialize`` field.  The ``description`` field is passed to the ``doc`` field in the Pyomo ``Param``.  Units are passed directly with an additional check.  All Constants set the Pyomo ``Param`` ``mutable`` field to True.

Non-scalar constants are constructed using Pyomo ``Sets``.  Sets are constructed to be integer sets that fill the entire interval from lower bound to upper bound, ie a vector constant of length 5 would create a Pyomo ``Set`` with valid indices [0,1,2,3,4] with no skips.  In this way, non-scalar constants are slightly less flexible than general non-scalar Pyomo ``Param``.


Examples
--------


A standard declaration statement
++++++++++++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constants_Snippet_01
    :end-before: # END: Constants_Snippet_01


Shortest possible declaration
+++++++++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constants_Snippet_02
    :end-before: # END: Constants_Snippet_02


An alternative units definition
+++++++++++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constants_Snippet_03
    :end-before: # END: Constants_Snippet_03


A vector constant
+++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constants_Snippet_04
    :end-before: # END: Constants_Snippet_04


A matrix/tensor constant
++++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constants_Snippet_05
    :end-before: # END: Constants_Snippet_05


More complicated units definition
+++++++++++++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constants_Snippet_06
    :end-before: # END: Constants_Snippet_06


Tips
----

* Declare constants in alphabetical order.  Trust me.  It's a pain at first, but it saves a huge amount of time down the road, especially for large models.
* Designate a section in your file for constant declarations, as is done in the :doc:`introductory example <./quickstart>`
* Align all of your constant declarations in a pretty, grid like fashion.  Depending on preference, these may or may not line up with variable declarations (I usually do not bother with this)
* Use the keyword names during constant declarations.  Takes extra space, but is a massive boost to readability and intrepretability
* Declare one constant on one single line with no breaks, no matter what style guides tell you.  Again, this is a significant boost to readability
* Do not skimp out on the description field, it is extremely helpful
