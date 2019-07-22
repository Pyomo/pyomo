Overview
======================================

Numerical optimization has proven to be an efficient tool for characterizing, designing, and operating chemical processes.  Efficient optimization algorithms are typically complex code bases written in low-level compiled programming languages.  However, this complexity also impedes development of new algorithms, requiring significant programming and software engineering expertise and a steep learning curve. This is especially true when developing decomposition approaches to solve large-scale structured problems in parallel.

To address and mitigate these challenges, we present PyNumero, a Python package for numerical optimization that provides a high-level programming framework for rapid development of nonlinear optimization algorithms. The package gives access to all high-level features of the Python programming language without making large sacrifices on computational performance. It combines the capabilities of the modeling language Pyomo with efficient libraries like the AMPL Solver Library (ASL), the Harwell Subroutine Library (HSL), the Message Passing Interface (mpi4py) and NumPy/SciPy. This combination makes PyNumero an excellent tool to develop numerical optimization algorithms that are interfaced with Pyomo. Furthermore, PyNumero performs all linear algebra operations in compiled code, and is designed to avoid marshalling of data between the C and Python environments, allowing for high-level development of algorithms without a significant sacrifice in performance.

Our goal with PyNumero is to make easier for Python users to write fast optimization algorithms. With this goal in mind we have implemented what we believe are the necessary building blocks required for writing any nonlinear optimization algorithm.
As we will see later we pay particular attention to decomposition algorithms that can exploit structure embedded within a variety of nonlinear optimization problem classes. The building blocks are presented below:

.. toctree::
   :maxdepth: 1

   amlpyomo
   nlpinterface
   sparsealgebra
   blockalgebra
   parallelalgebra
   linearsolvers
   algorithms

Our emphasis on designing a flexible framework is in line with recent developments in Julia with the package `Smooth Optimizers <https://juliasmoothoptimizers.github.io/>`_. PyNumero's goal is somewhat different from Julia Smooth Optimizers as it not only offers tools for general nonlinear optimization, but it provides a framework for experimenting with decomposition algorithms. Unlike Julia, Python offers complete support for object-oriented software. PyNumero uses object-oriented principles comprehensively, applying them to linear algebra interfaces and problem formulations that exploit block-structures via polymorphism and inheritance mechanisms.

https://aiche.confex.com/aiche/2018/meetingapp.cgi/Session/38903
