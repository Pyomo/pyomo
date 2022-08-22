from collections import namedtuple
from pyomo.common.timing import HierarchicalTimer


TimeBins = namedtuple("TimeBins", ["set_primals", "constraints", "jacobian"])
timebins = TimeBins("set_primals", "constraints", "jacobian")


class _SquareNlpSolverBase(object):
    """A base class for NLP solvers that act on a square system
    of equality constraints.

    """

    def __init__(self, nlp, timer=None):
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
        self._timer = timer
        self._nlp = nlp

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
        self._timer.start(timebins.set_primals)
        self._nlp.set_primals(x0)
        self._timer.stop(timebins.set_primals)

        self._timer.start(timebins.constraints)
        values = self._nlp.evaluate_eq_constraints()
        self._timer.stop(timebins.constraints)

        return values

    def evaluate_jacobian(self, x0):
        # NOTE: NLP object should handle any caching
        self._timer.start(timebins.set_primals)
        self._nlp.set_primals(x0)
        self._timer.stop(timebins.set_primals)

        self._timer.start(timebins.jacobian)
        jac = self._nlp.evaluate_jacobian_eq()
        self._timer.stop(timebins.jacobian)

        return jac


class DenseSquareNlpSolver(_SquareNlpSolverBase):
    """A square NLP solver that uses a dense Jacobian
    """

    def evaluate_jacobian(self, x0):
        sparse_jac = super().evaluate_jacobian(x0)

        # Should sparse->dense be timed separately?
        self._timer.start(timebins.jacobian)
        dense_jac = sparse_jac.toarray()
        self._timer.stop(timebins.jacobian)

        return dense_jac
