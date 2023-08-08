Constraints
===========


Overview
--------

Constraints are the mathematical representation of rules that are imposed on your decisions/variables when minimizing or maximizing.  In engineering design, constraints are often imposed by physics or operational limits.

The Constraint constructor is a very thin wrapper on pyomo ``Constraint``, and so experienced pyomo users will not see any significant differences from base pyomo.  


Construction
------------

Constraints are constructed by creating an instance of a new constraint in a EDI Formulation 

.. py:function:: f.Constraint(expr)

    Declares a constraint in a pyomo.edi.formulation

   :param expr: The expression representing the constraint
   :type  expr: pyomo expression

   :return: None
   :rtype: None


However, the expected use case is the ``f.ConstraintList()`` function:

.. py:function:: f.ConstraintList(conList)

    Declares new constraints in a pyomo.edi.formulation from a list of inputs

   :param conList: The list of constraints to be generated.  Entries will be pyomo expressions, or lists/tuples/dicts that are used to create RuntimeConstraints (see :doc:`here <./blackboxconstraints>`)
   :type  conList: list

   :return: None
   :rtype: None


Relation to Pyomo Constraint
----------------------------

The EDI constraint constructor is essentially a direct pass through to base pyomo.  Constraints will be added to the ``pyomo.ConcreteModel`` in increasing order with key ``constraint_###`` where the the index of the objective appears after the underscore.  First constraint is labeled as ``constraint_1``, and constraint names are never padded with zeros.  RuntimeConstraints also contribute to this counter.


Examples
--------

A standard declaration statement
++++++++++++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constraints_Snippet_01
    :end-before: # END: Constraints_Snippet_01


With the core constructor
+++++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constraints_Snippet_02
    :end-before: # END: Constraints_Snippet_02

Using indexed variables and constants
+++++++++++++++++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constraints_Snippet_03
    :end-before: # END: Constraints_Snippet_03

Tips
----

* For the typical user, constraints should always be constructed using the ``f.ConstraintList()`` function to produce a cleaner input file that is easier to modify


Known Issues
------------

* Indexed variables must be broken up using either indices or a pyomo rule (see `this issue <https://github.com/codykarcher/pyomo/issues/3>`__)
* Units that are inconsistent, but not the same (ie, meters and feet) will flag as invalid when checking units (see `this issue <https://github.com/codykarcher/pyomo/issues/6>`__)
