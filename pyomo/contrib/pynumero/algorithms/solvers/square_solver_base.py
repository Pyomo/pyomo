class _SquareNlpSolverBase(object):

    def __init__(self, nlp):
        self._nlp = nlp
        # TODO: Validate that NLP is a square problem

    def solve(self, x0=None):
        # the NLP has a natural initial guess - the cached primal
        # values. x0 may be provided if a different initial guess
        # is desired.
        raise NotImplementedError(
            "%s has not implemented the solve method"
            % self.__class__
        )

    def evaluate_function(self, x0):
        self._nlp.set_primals(x0)
        return self._nlp.evaluate_constraints()

    def evaluate_jacobian(self, x0):
        self._nlp.set_primals(x0)
        return self._nlp.evaluate_jacobian()
