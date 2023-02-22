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

We are working on a journal article about Incidence Analysis and the underlying
methods. In the meantime, if you use Incidence Analysis in your research, you
may cite the following conference paper:

.. code-block:: bibtex

    @inproceedings{Parker2023Dulmage,
      title={{An application of the Dulmage-Mendelsohn partition to the analysis of a discretized dynamic chemical looping combustion reactor model}},
      author={Robert Parker and Chinedu Okoli and Bethany Nicholson and John Siirola and Lorenz Biegler},
      booktitle={Proceedings of FOCAPO/CPC 2023},
      year={2023}
    }
