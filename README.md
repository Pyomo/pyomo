[![Github Actions Status](https://github.com/Pyomo/pyomo/workflows/GitHub%20CI/badge.svg?event=push)](https://github.com/Pyomo/pyomo/actions?query=event%3Apush+workflow%3A%22GitHub+CI%22)
[![Jenkins Status](https://img.shields.io/jenkins/s/https/software.sandia.gov/downloads/pub/pyomo/jenkins/Pyomo_trunk.svg?logo=jenkins&logoColor=white)](https://jenkins-srn.sandia.gov/job/Pyomo_trunk)
[![codecov](https://codecov.io/gh/Pyomo/pyomo/branch/master/graph/badge.svg)](https://codecov.io/gh/Pyomo/pyomo)
[![Documentation Status](https://readthedocs.org/projects/pyomo/badge/?version=latest)](http://pyomo.readthedocs.org/en/latest/)

[![GitHub contributors](https://img.shields.io/github/contributors/pyomo/pyomo.svg)](https://github.com/pyomo/pyomo/graphs/contributors)
[![Merged PRs](https://img.shields.io/github/issues-pr-closed-raw/pyomo/pyomo.svg?label=merged+PRs)](https://github.com/pyomo/pyomo/pulls?q=is:pr+is:merged)
[![Issue stats](http://isitmaintained.com/badge/resolution/pyomo/pyomo.svg)](http://isitmaintained.com/project/pyomo/pyomo)
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)

[![a COIN-OR project](https://www.coin-or.org/GitHub/coin-or-badge.png)](https://www.coin-or.org)

## Pyomo Overview
Pyomo is a Python-based open-source software package that supports a diverse set of optimization capabilities for formulating and analyzing optimization models. Pyomo can be used to define symbolic problems, create concrete problem instances, and solve these instances with standard solvers. Pyomo supports a wide range of problem types, including:

 -  Linear programming
 -  Quadratic programming
 -  Nonlinear programming
 -  Mixed-integer linear programming
 -  Mixed-integer quadratic programming
 -  Mixed-integer nonlinear programming
 -  Mixed-integer stochastic programming
 -  Generalized disjunctive programming
 -  Differential algebraic equations
 -  Mathematical programming with equilibrium constraints

Pyomo supports analysis and scripting within a full-featured programming language. Further, Pyomo has also proven an effective framework for developing high-level optimization and analysis tools.  For example, the PySP package provides generic solvers for stochastic programming. PySP leverages the fact that Pyomo's modeling objects are embedded within a full-featured high-level programming language, which allows for transparent parallelization of subproblems using Python parallel communication libraries.

* [Pyomo Home](http://www.pyomo.org)
* [About Pyomo](http://www.pyomo.org/about)
* [Download](http://www.pyomo.org/installation/)
* [Documentation](http://www.pyomo.org/documentation/)
* [Performance Plots](https://software.sandia.gov/downloads/pub/pyomo/performance/index.html)
* [Blog](http://www.pyomo.org/blog/)

Pyomo was formerly released as the Coopr software library.

Pyomo is available under the BSD License, see the LICENSE.txt file.

Pyomo is currently tested with the following Python implementations:

* CPython: 2.7, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9
* PyPy: 2, 3

### Installation

#### PyPI [![PyPI version](https://img.shields.io/pypi/v/pyomo.svg?maxAge=3600)](https://pypi.org/project/Pyomo/) [![PyPI downloads](https://img.shields.io/pypi/dm/pyomo.svg?maxAge=21600)](https://pypistats.org/packages/pyomo)

    pip install pyomo

#### Anaconda [![Anaconda version](https://anaconda.org/conda-forge/pyomo/badges/version.svg)](https://anaconda.org/conda-forge/pyomo) [![Anaconda downloads](https://anaconda.org/conda-forge/pyomo/badges/downloads.svg)](https://anaconda.org/conda-forge/pyomo)

    conda install -c conda-forge pyomo

### Tutorials and Examples

* [Pyomo Workshop Slides](https://software.sandia.gov/downloads/pub/pyomo/Pyomo-Workshop-Summer-2018.pdf)
* [Prof. Jeffrey Kantor's Pyomo Cookbook](https://jckantor.github.io/ND-Pyomo-Cookbook/)
* [Pyomo Gallery](https://github.com/Pyomo/PyomoGallery)

### Getting Help

* [Ask a Pyomo Question on StackExchange](https://stackoverflow.com/questions/ask?tags=pyomo)
* [Pyomo Forum](https://groups.google.com/forum/?hl=en#!forum/pyomo-forum)
* [Add a Ticket](https://github.com/Pyomo/pyomo/issues/new)
* [Find a Ticket](https://github.com/Pyomo/pyomo/issues) and **Vote On It**!

### Developers

Pyomo development moved to this repository in June, 2016 from
Sandia National Laboratories. Developer discussions are hosted by [google groups](https://groups.google.com/forum/#!forum/pyomo-developers).

By contributing to this software project, you are agreeing to the following terms and conditions for your contributions:

1. You agree your contributions are submitted under the BSD license. 
2. You represent you are authorized to make the contributions and grant the license. If your employer has rights to intellectual property that includes your contributions, you represent that you have received permission to make contributions and grant the required license on behalf of that employer. 
