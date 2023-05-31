We are pleased to announce the release of Pyomo 6.6.1.

Pyomo is a collection of Python software packages that supports a
diverse set of optimization capabilities for formulating and analyzing
optimization models.

The following are highlights of the 6.0 release series:

 - Improved stability and robustness of core Pyomo code and solver interfaces
 - Integration of Boolean variables into GDP
 - Integration of NumPy support into the Pyomo expression system
 - Implemented a more performant and robust expression generation system
 - Implemented a more performant NL file writer (NLv2)
 - Implemented a more performant LP file writer (LPv2)
 - Applied [PEP8 standards](https://peps.python.org/pep-0008/) throughout the
   codebase
 - Added support for Python 3.10, 3.11
 - Removed support for Python 3.6
 - Removed the `pyomo check` command
 - New packages:
    - APPSI (Auto-Persistent Pyomo Solver Interfaces)
    - CP (Constraint programming models and solver interfaces)
    - DoE (Model based design of experiments)
    - External grey box models
    - IIS (Standard interface to solver IIS capabilities)
    - MPC (Data structures/utils for rolling horizon dynamic optimization)
    - piecewise (Modeling with and reformulating multivariate piecewise linear
      functions)
    - PyROS (Pyomo Robust Optimization Solver)
    - Structural model analysis
    - Rewrite of the TrustRegion Solver

A full list of updates and changes is available in the
[`CHANGELOG.md`](https://github.com/Pyomo/pyomo/blob/main/CHANGELOG.md).

Enjoy!

 - Pyomo Developer Team
 - pyomo-developers@googlegroups.com
 - https://www.pyomo.org


About Pyomo
-----------

The Pyomo home page provides resources for Pyomo users:

 * https://www.pyomo.org

Detailed documentation is hosted on Read the Docs:

 * https://pyomo.readthedocs.io/en/stable/

Pyomo development is hosted at GitHub:

 * https://github.com/Pyomo

Get help at:

 * StackOverflow: https://stackoverflow.com/questions/tagged/pyomo
 * Pyomo Forum:   https://groups.google.com/group/pyomo-forum/
