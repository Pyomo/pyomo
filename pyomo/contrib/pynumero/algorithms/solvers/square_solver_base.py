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


# This is like a DirectSolver or PersistentSolver, but with a more limited API.
# It may make sense to inherit from one of these classes at some point.
# It may also make sense to merge this with the sensitivity toolbox class,
# so that sensitivities wrt this model may be calculated. However, for this
# solve, we limit ourselves to square problems, and don't immediately require
# derivatives.
# The purpose of this class is to define an API that we can use for implicit
# function solves. The reason this API is useful is that it allows resolves
# with different parameters without rewriting the NL file.
class ParameterizedSquareSolver(object):
    """
    Given a square Pyomo model representing a system:
    g(x, y) = 0
    where x are variables and y are parameters.
    This class allows updating parameters and solving for variables.

    """

    def __init__(self, model, param_vars):
        # Is param_vars a confusing name? These must be variables, but they
        # will be treated as paramters
        # In a proper Pyomo interface, these could be actual parameters
        # or variables. Maybe this should even go through the existing
        # sensitivity interface.
        self._model = model
        self._param_vars = param_vars

    def update_parameters(self, values):
        raise NotImplementedError()

    def solve(self):
        raise NotImplementedError()


class SquareNlpSolverBase(object):
    """A base class for NLP solvers that act on a square system
    of equality constraints.

    """
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
                " equalities."
                % (self._nlp.n_primals(), self._nlp.n_eq_constraints())
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
        self._timer.start("eval_f")
        self._nlp.set_primals(x0)
        values = self._nlp.evaluate_eq_constraints()
        self._timer.stop("eval_f")
        return values

    def evaluate_jacobian(self, x0):
        # NOTE: NLP object should handle any caching
        self._timer.start("eval_j")
        self._nlp.set_primals(x0)
        self._jacobian = self._nlp.evaluate_jacobian_eq(out=self._jacobian)
        self._timer.stop("eval_j")
        return self._jacobian


class DenseSquareNlpSolver(SquareNlpSolverBase):
    """A square NLP solver that uses a dense Jacobian
    """

    def evaluate_jacobian(self, x0):
        sparse_jac = super().evaluate_jacobian(x0)
        dense_jac = sparse_jac.toarray()
        return dense_jac
