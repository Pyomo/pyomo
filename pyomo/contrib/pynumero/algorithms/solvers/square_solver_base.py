class _SquareNlpSolverBase(object):
    """A base class for NLP solvers that act on a square system
    of equality constraints.

    """

    def __init__(self, nlp):
        """
        Arguments
        ---------
        nlp: ExtendedNLP
            An instance of ExtendedNLP that will be solved.
            ExtendedNLP is required to ensure that the NLP has equal
            numbers of primal variables and equality constraints.

        """
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
        self._nlp.set_primals(x0)
        return self._nlp.evaluate_constraints()

    def evaluate_jacobian(self, x0):
        # NOTE: NLP object should handle any caching
        self._nlp.set_primals(x0)
        return self._nlp.evaluate_jacobian()
