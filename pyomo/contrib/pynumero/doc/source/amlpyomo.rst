Modeling Language
=================

Optimization problems of complex systems are frequently encountered in applied mathematics and engineering applications. Finding an optimal solution for such problems is particularly challenging because of several reasons. First, complex optimization problems are usually large-scale problems with numerous integer and continuous variables, and linear and nonlinear expressions. This complexities do not only complicate the solution but also the construction of the optimization formulation itself. Fortunately, the introduction of algebraic modeling languages (AMLs) has allowed for the rapid creation of complex optimization problems by reducing the burden of model development. This eases the construction of complicated optimization problems, and also makes it easier to construct intractably large optimization formulations (in a flexible, consistent manner).

Our project builds uppon the algebraic modeling features of the Python package Pyomo and allows optimization practitioners to solve models in python. Different than the standard AML/Solver dynamic in which the model is built at the AML level and solved at the solver level, we explore with PyNumero a unified modeling and solution platform.  

.. _fig-example1:
.. figure:: figures/PyNumeroOverview.png
   :alt: Summary of Pyomo Features 

The idea is to facilitate research of nonlinear optimization algorithms by allowing Pyomo users to write taylored algorithms for solving their problems of interest instead of solving them with off-the-shelf optimization solvers. Our experience tells us that many real-world, large-scale optimization that have been implemented in Pyomo typically require specialized solution approaches that exploit knowledge of the application modeled. Decomposition-based algorithms are one such approach for solving these challenging problems. Combining PyNumero with Pyomo we seek to facilitate the implementation of this approaches.

For users that are new to Pyomo as well to PyNumero, we encourage them to read the documentation of Pyomo before getting into the details of PyNumero

`Pyomo Docs <https://pyomo.readthedocs.io/en/latest/>`_
