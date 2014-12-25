============
Pyomo README
============

Pyomo is a Python-based open-source software package that supports a diverse set of optimization capabilities for formulating and analyzing optimization models.

Modeling optimization applications is a core capability of Pyomo.  Pyomo can be used to define symbolic problems, create concrete problem instances, and solve these instances with standard solvers.  Thus, Pyomo provides a capability that is commonly associated with algebraic modeling languages such as AMPL, AIMMS, and GAMS, but Pyomo's modeling objects are embedded within a full-featured high-level programming language with a rich set of supporting libraries.  Pyomo supports a wide range of problem types, including:

 -  Linear programming
 -  Quadratic programming
 -  Nonlinear programming
 -  Mixed-integer linear programming
 -  Mixed-integer quadratic programming
 -  Mixed-integer nonlinear programming
 -  Mixed-integer stochastic programming
 -  Generalized disjunctive programming
 -  Differential algebraic equations
 -  Bilevel programming
 -  Mathematical programming with equilibrium constraints

Pyomo supports analysis and scripting within a full-featured programming language.  Further, Pyomo has also proven an effective framework for developing high-level optimization and analysis tools.  For example, the PySP package provides generic solvers for stochastic programming.  PySP leverages the fact that Pyomo's modeling objects are embedded within a full-featured high-level programming language, which allows for transparent parallelization of subproblems using Python parallel communication libraries.

Pyomo was formerly released as the Coopr software library.


-------
License
-------

BSD.  See the LICENSE.txt file.


------------
Organization
------------

+ Directories

  * pyomo - The root directory for Pyomo source code

+ Documentation and Bug Tracking

  * Trac wiki: https://software.sandia.gov/trac/pyomo

+ Authors

  * See the AUTHORS.txt file.

+ Mailing List

  * pyomo-forum@googlegroups.com
    - The main list for help and announcements
  * pyomo-developers@googlegroups.com
    - Where developers of Pyomo discuss new features

--------------------
Third Party Software
--------------------

The following software is bundled with Pyomo, and it release under BSD-compatible licenses:

. pyomo/scripts/pyomo_install

    https://github.com/pypa/pip/blob/develop/LICENSE.txt

