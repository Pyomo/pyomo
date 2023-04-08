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

from collections import namedtuple
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigBlock
from pyomo.util.subsystems import create_subsystem_block


class SquareNlpSolverBase(object):
    """A base class for NLP solvers that act on a square system
    of equality constraints.

    """

    # Ideally, this ConfigBlock would contain options that are valid for any
    # square NLP solver. However, no such options seem to exist while
    # preserving the names of the SciPy function arguments. E.g., tolerance
    # is "tol" in some solvers and "xtol" in others, and some solvers
    # support "maxiter" while others support "maxfev". It may be useful to
    # attempt some standardization by, e.g., mapping tol->xtol, then
    # specifying "universal" options here, but this can happen at a later
    # date as these solvers see more use.
    OPTIONS = ConfigBlock()

    def __init__(self, nlp, timer=None, options=None):
        """
        Arguments
        ---------
        nlp: ExtendedNLP
            An instance of ExtendedNLP that will be solved.
            ExtendedNLP is required to ensure that the NLP has equal
            numbers of primal variables and equality constraints.

        """
        if timer is None:
            timer = HierarchicalTimer()
        if options is None:
            options = {}
        self.options = self.OPTIONS(options)

        self._timer = timer
        self._nlp = nlp
        self._function_values = None
        self._jacobian = None

        if self._nlp.n_eq_constraints() != self._nlp.n_primals():
            raise RuntimeError(
                "Cannot construct a square solver for an NLP that"
                " does not have the same numbers of variables as"
                " equality constraints. Got %s variables and %s"
                " equalities." % (self._nlp.n_primals(), self._nlp.n_eq_constraints())
            )
        # Checking for a square system of equalities is easy, but checking
        # bounds is a little difficult. We don't know how an NLP will
        # implement bounds (no bound could be None, np.nan, or np.inf),
        # so it is annoying to check that bounds are not present.
        # Instead, we just ignore bounds, and the user must know that the
        # result of this solver is not guaranteed to respect bounds.
        # While it is easier to check that inequalities are absent,
        # for consistency, we take the same approach and simply ignore
        # them.

    def solve(self, x0=None):
        # the NLP has a natural initial guess - the cached primal
        # values. x0 may be provided if a different initial guess
        # is desired.
        raise NotImplementedError(
            "%s has not implemented the solve method" % self.__class__
        )

    def evaluate_function(self, x0):
        # NOTE: NLP object should handle any caching
        self._nlp.set_primals(x0)
        values = self._nlp.evaluate_eq_constraints()
        return values

    def evaluate_jacobian(self, x0):
        # NOTE: NLP object should handle any caching
        self._nlp.set_primals(x0)
        self._jacobian = self._nlp.evaluate_jacobian_eq(out=self._jacobian)
        return self._jacobian


class DenseSquareNlpSolver(SquareNlpSolverBase):
    """A square NLP solver that uses a dense Jacobian"""

    def evaluate_jacobian(self, x0):
        sparse_jac = super().evaluate_jacobian(x0)
        dense_jac = sparse_jac.toarray()
        return dense_jac


class ScalarDenseSquareNlpSolver(DenseSquareNlpSolver):
    # A base class for solvers for scalar equations.
    # Not intended to be instantiated directly. Instead,
    # NewtonNlpSolver or SecantNewtonNlpSolver should be used.

    def __init__(self, nlp, timer=None, options=None):
        super().__init__(nlp, timer=timer, options=options)
        if nlp.n_primals() != 1:
            raise RuntimeError(
                "Cannot use the scipy.optimize.newton solver on an NLP with"
                " more than one variable and equality constraint. Got %s"
                " primals. Please use RootNlpSolver or FsolveNlpSolver instead."
            )
