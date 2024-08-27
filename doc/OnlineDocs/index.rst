Pyomo Documentation |release|
=============================

.. image:: /../logos/pyomo/PyomoNewBlue3.png
   :scale: 10%
   :align: right

Pyomo is a Python-based, open-source optimization modeling language
with a diverse set of optimization capabilities.


.. list-table::
   :width: 100%
   :class: index-table

   * - Getting Started
         | :doc:`Installation <getting_started/installation>`
         | :doc:`Pyomo Overview <getting_started/pyomo_overview/index>`
     - How-To Guide
         | :doc:`Interrogating Models`
         | :doc:`Manipulating Models`
         | :doc:`Solver Recipes`
         | :doc:`Debugging Models`
         | :doc:`Contributing to Pyomo`
   * - User Explanations
         | :doc:`Pyomo Philosophy`
         |    :doc:`Concrete and Abstract Models`
         |    :doc:`Component Hierarchy`
         |    :doc:`Expression System`
         |    :doc:`Transformations`
         | :doc:`Modeling in Pyomo`
         |    :doc:`Math Programming`
         |    :doc:`GDP`
         |    :doc:`DAE`
         |    :doc:`Network`
         |    :doc:`Piecewise Linear`
         |    :doc:`Constraint Programming`
         |    :doc:`Units of Measure`
         | :doc:`Solvers`
         |    :doc:`PyROS`
         |    :doc:`MindtPy`
         |    :doc:`Trust Region`
         |    :doc:`Pynumero`
         | :doc:`Analysis in Pyomo`
         |    :doc:`IIS`
         |    :doc:`FBBT`
         |    :doc:`Incidence Analysis`
         |    :doc:`Parameter Estimation`
         |    :doc:`Design of Experiments`
         |    :doc:`MPC`
         |    :doc:`AOS`
         | :doc:`Modeling Utilities`
         |    :doc:`Latex Printer`
         |    :doc:`FME`
         |    :doc:`Model Viewer`
         |    :doc:`Model Flattening`
         | :doc:`Developer Utilities`
         |    :doc:`Configuration System`
         |    :doc:`Deprecation System`
         | :doc:`Experimental`
         |    :doc:`Kernel`
     - Reference Guide
         | :doc:`Library Reference <reference_guide/index>`
         | :doc:`Common Warnings and Errors`
         | :doc:`Preview capabilities through ``pyomo.__future__`` <developer_guide/future>`

.. toctree::
   :hidden:
   :maxdepth: 2

    Getting Started <getting_started/index>
    Hot-To Guides <how_to_guide/index>
    User Explanations <user_explanations/index>
    Reference Guide <reference_guide/index>


Pyomo Resources
---------------

Pyomo development is hosted at GitHub:

* https://github.com/Pyomo/pyomo

See the Pyomo Forum for online discussions of Pyomo or to ask a question:

* http://groups.google.com/group/pyomo-forum/

Ask a question on StackOverflow using the `#pyomo` tag:

* https://stackoverflow.com/questions/ask?tags=pyomo 

Additional Pyomo tutorials and examples can be found at the following links:

* `Pyomo — Optimization Modeling in Python
  <https://link.springer.com/book/10.1007/978-3-030-68928-5>`_ ([PyomoBookIII]_)

* `Pyomo Workshop Slides and Exercises
  <https://github.com/Pyomo/pyomo-tutorials>`_

* `Prof. Jeffrey Kantor's Pyomo Cookbook
  <https://jckantor.github.io/ND-Pyomo-Cookbook/>`_

* The `companion notebooks <https://mobook.github.io/MO-book/intro.html>`_
  for *Hands-On Mathematical Optimization with Python*

* `Pyomo Gallery <https://github.com/Pyomo/PyomoGallery>`_


Contributing to Pyomo
---------------------

Interested in contributing code or documentation to the project? Check out our
:doc:`Contribution Guide <contribution_guide>`

Related Packages
----------------

Pyomo is a key dependency for a number of other software packages for
specific domains or customized solution strategies. A non-comprehensive
list of Pyomo-related packages may be found :doc:`here <related_packages>`.


Citing Pyomo
------------

Bynum, Michael L., Gabriel A. Hackebeil, William E. Hart, Carl D. Laird, Bethany L. Nicholson, John D. Siirola, Jean-Paul Watson, and David L. Woodruff. Pyomo - Optimization Modeling in Python, 3rd Edition. Springer, 2021.

Hart, William E., Jean-Paul Watson, and David L. Woodruff. "Pyomo: modeling and solving mathematical programs in Python." Mathematical Programming Computation 3, no. 3 (2011): 219-260.

