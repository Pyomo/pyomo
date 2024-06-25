Objectives
==========

Overview
--------

Objectives are the mathematical representation of what you wish to minimize or maximize.  In engineering design, objectives are often minimizing cost, material, or time, or alternatively maximizing profit or utility.

The Objective constructor is a very thin wrapper on pyomo ``Objective``, and so experienced pyomo users will not see any significant differences from base pyomo.  


Construction
------------

Objectives are constructed by creating an instance of a new objective in a EDI Formulation 

.. py:function:: f.Objective(expr, sense=minimize)

    Declares an objective in a pyomo.edi.formulation

   :param expr: The expression to be optimized
   :type  expr: pyomo expression
   :param sense: The sense in which the objective should be optimized, either minimized or maximized.  Can import ``minimize`` and ``maximize`` from ``pyomo.environ``, but minimize corresponds to an integer of 1 and maximize to an integer of -1.
   :type  sense: int

   :return: None
   :rtype: None


Relation to Pyomo Objective
---------------------------

The EDI objective constructor is essentially a direct pass through to base pyomo.  Objectives will be added to the ``pyomo.ConcreteModel`` in increasing order with key ``objective_###`` where the the index of the objective appears after the underscore.  First objective is labeled as ``objective_1``, and objective names are never padded with zeros.


Examples
--------

A standard declaration statement
++++++++++++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Objectives_Snippet_01
    :end-before: # END: Objectives_Snippet_01

With a non-linear objective
+++++++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Objectives_Snippet_02
    :end-before: # END: Objectives_Snippet_02

Explicitly minimize
+++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Objectives_Snippet_03
    :end-before: # END: Objectives_Snippet_03

Explicitly minimize using integer
+++++++++++++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Objectives_Snippet_04
    :end-before: # END: Objectives_Snippet_04


Maximizing
++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Objectives_Snippet_05
    :end-before: # END: Objectives_Snippet_05

Maximizing using integer
++++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Objectives_Snippet_06
    :end-before: # END: Objectives_Snippet_06

Using indexed variables and constants
+++++++++++++++++++++++++++++++++++++

.. literalinclude:: ../../../../pyomo/contrib/edi/tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Objectives_Snippet_07
    :end-before: # END: Objectives_Snippet_07


Tips
----

* Objectives are a pretty natural place to break your file.  Put at least one blank line above and below the objective constructor and use good sectioning to create a whitespace easily identifiable when scrolling quickly 
