Overview
======================================

Numerical optimization has proven to be an efficient tool for characterizing, designing, and operating chemical processes.  Efficient optimization algorithms are typically complex code bases written in low-level compiled programming languages.  However, this complexity also impedes development of new algorithms, requiring significant programming and software engineering expertise and a steep learning curve. This is especially true when developing decomposition approaches to solve large-scale structured problems in parallel.

To address and mitigate these challenges, we present Pynumero, a Python package for numerical optimization that provides a high-level programming framework for rapid development of nonlinear optimization algorithms. The package gives access to all high-level features of the Python programming language without making large sacrifices on computational performance. It combines the capabilities of the modeling language Pyomo with efficient libraries like the AMPL Solver Library (ASL), the Harwell Subroutine Library (HSL), the Message Passing Interface (mpi4py) and NumPy/SciPy. This combination makes Pynumero an excellent tool to develop numerical optimization algorithms that are interfaced with Pyomo. Furthermore, Pynumero performs all linear algebra operations in compiled code, and is designed to avoid marshalling of data between the C and Python environments, allowing for high-level development of algorithms without a significant sacrifice in performance.

Our goal with PyNumero is to make easier for Python users to write optimization algorithms. With this goal in mind we have implemented what we believed are the necessary building blocks required for writing any nonlinear optimization algorithm.
We pay particular attention to decomposition algorithms that can exploit structure embedded within a variaety of nonlinear optimization problem classes. The building blocks are presented below:

.. toctree::
   :maxdepth: 1

   amlpyomo
   nlpinterface
   sparsealgebra
   blockalgebra
   linearsolvers
   
https://aiche.confex.com/aiche/2018/meetingapp.cgi/Session/38903
