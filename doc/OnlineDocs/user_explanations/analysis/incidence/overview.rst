.. _incidence_overview:

Overview
========

What is Incidence Analysis?
---------------------------

A Pyomo extension for constructing the bipartite incidence graph of variables
and constraints, and an interface to useful algorithms for analyzing or
decomposing this graph.

Why is Incidence Analysis useful?
---------------------------------

It can identify the source of certain types of singularities in a system of
variables and constraints. These singularities often violate assumptions made
while modeling a physical system or assumptions required for an optimization
solver to guarantee convergence. In particular, interior point methods used for
nonlinear local optimization require the Jacobian of equality constraints (and
active inequalities) to be full row rank, and this package implements the
Dulmage-Mendelsohn partition, which can be used to determine if this Jacobian
is structurally rank-deficient.

Who develops and maintains Incidence Analysis?
----------------------------------------------

This extension was developed by Robert Parker while a PhD student in
Professor Biegler's lab at Carnegie Mellon University, with guidance
from Bethany Nicholson and John Siirola at Sandia.

How can I cite Incidence Analysis?
----------------------------------

If you use Incidence Analysis in your research, we would appreciate you citing
the following paper:

.. code-block:: bibtex

    @article{parker2023dulmage,
    title = {Applications of the {Dulmage-Mendelsohn} decomposition for debugging nonlinear optimization problems},
    journal = {Computers \& Chemical Engineering},
    volume = {178},
    pages = {108383},
    year = {2023},
    issn = {0098-1354},
    doi = {https://doi.org/10.1016/j.compchemeng.2023.108383},
    url = {https://www.sciencedirect.com/science/article/pii/S0098135423002533},
    author = {Robert B. Parker and Bethany L. Nicholson and John D. Siirola and Lorenz T. Biegler},
    }
