#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.extensions import ExtensionBuilderFactory
from pyomo.opt import SolverFactory
from .build import PyNumeroBuilder
from .algorithms.solvers.cyipopt_solver import PyomoCyIpoptSolver
from .algorithms.solvers.scipy_solvers import (
    PyomoFsolveSolver,
    PyomoRootSolver,
    PyomoNewtonSolver,
    PyomoSecantNewtonSolver,
)


def load():
    ExtensionBuilderFactory.register('pynumero')(PyNumeroBuilder)
    SolverFactory.register(
        'cyipopt', doc='Cyipopt: direct python bindings to the Ipopt NLP solver'
    )(PyomoCyIpoptSolver)
    SolverFactory.register(
        "scipy.fsolve",
        doc=("fsolve: A SciPy wrapper around MINPACK's hybrd and hybrj algorithms"),
    )(PyomoFsolveSolver)
    SolverFactory.register(
        "scipy.root", doc=("root: Find the root of a vector function")
    )(PyomoRootSolver)
    SolverFactory.register(
        "scipy.newton", doc="newton: Find a zero of a scalar-valued function"
    )(PyomoNewtonSolver)
    SolverFactory.register(
        "scipy.secant-newton",
        doc=(
            "secant-newton: Take a few secant iterations to try to converge"
            " a potentially linear equation quickly, then switch to Newton's"
            " method"
        ),
    )(PyomoSecantNewtonSolver)
