Modeling Language
=================

Optimization problems of complex systems are frequently encountered in applied mathematics and engineering applications. Finding an optimal solution for such problems is particularly challenging because of several reasons. First, complex optimization problems are usually large-scale problems with numerous integer and continuous variables, and linear and nonlinear expressions. This complexities do not only complicate the solution but also the construction of the optimization formulation itself. Fortunately, the introduction of algebraic modeling languages (AMLs) has allowed for the rapid creation of complex optimization problems by reducing the burden of model development. This eases the construction of complicated optimization problems, and also makes it easier to construct intractably large optimization formulations (in a flexible, consistent manner).

Our project builds uppon the algebraic modeling features of the Python package Pyomo and allows optimization practitioners to solve models in Python. Different than the standard AML/Solver strategy in which the model is built at the AML level and solved at the solver level, we explore with PyNumero a unified modeling and solution platform.  

.. _fig-example1:
.. figure:: figures/PyNumeroOverview.png
   :alt: Summary of Pyomo Features 

The idea is to provide a framework for Pyomo users to write taylored algorithms for solving their problems of interest instead of solving them with off-the-shelf optimization solvers. Our experience has tought us that many real-world models implemented in Pyomo often require specialized solution approaches that exploit knowledge of the application modeled within the solution approach. Decomposition-based algorithms are one such approach for solving these challenging problems. Combining PyNumero with Pyomo users can directly interrogate Pyomo model features while writing their algorithm in Python.

The proposed workflow to solve an optimization problem with PyNumero consist on first writing an optimization model in `Pyomo <https://pyomo.readthedocs.io/en/latest/>`_, then translate the model into an NLP representation that performs efficiently function evaluations necessary during the execution of any solution algorithm. The NLP representation is implemented in Python and uses Numpy and Scipy together with the AMPL solver library. This combination allows for fast computation of NLP functions that are easily integrated into algorithm subroutines written in Python. 

In the next section we explain how to transform a Pyomo model into a PyNumero NLP.
