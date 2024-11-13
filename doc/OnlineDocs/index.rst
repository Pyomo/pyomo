=============================
Pyomo Documentation |release|
=============================

About Pyomo
-----------

.. image:: /../logos/pyomo/PyomoNewBlue3.png
   :scale: 10%
   :align: right

Pyomo is a Python-based open-source software package that supports a
diverse set of optimization capabilities for formulating, solving, and
analyzing optimization models.

A core capability of Pyomo is modeling structured optimization
applications.  Pyomo can be used to define general symbolic problems,
create specific problem instances, and solve these instances using
commercial and open-source solvers.


Contents
--------
.. list-table::
   :width: 100%
   :class: diataxis

   * - .. toctree::
          :maxdepth: 2
          :titlesonly:

          getting_started/index
     - .. toctree::
          :maxdepth: 2
          :titlesonly:

          howto/index
   * - .. toctree::
          :maxdepth: 3
          :titlesonly:

          explanation/index
     - .. toctree::
          :maxdepth: 3
          :titlesonly:

          reference/index

..
   toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   genindex
   modindex


Pyomo Resources
---------------

Pyomo development is hosted at GitHub:

* https://github.com/Pyomo/pyomo

See the Pyomo Forum for online discussions of Pyomo or to ask a question:

* http://groups.google.com/group/pyomo-forum/

Ask a question on StackOverflow using the ``#pyomo`` tag:

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

If you use Pyomo in your work, please cite:

    Bynum, Michael L., Gabriel A. Hackebeil, William E. Hart, Carl D. Laird,
    Bethany L. Nicholson, John D. Siirola, Jean-Paul Watson, and
    David L. Woodruff. Pyomo - Optimization Modeling in Python, 3rd
    Edition. Springer, 2021.

Additionally, several Pyomo capabilities and subpackages are described
in further detail in separate :ref:`publications`.
