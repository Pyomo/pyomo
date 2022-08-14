class _SquareNlpSolverBase(object):

    def __init__(self, nlp):
        self._nlp = nlp

    def solve(self, x0=None):
        # the NLP has a natural initial guess - the cached primal
        # values. x0 may be provided if a different initial guess
        # is desired.
        raise NotImplementedError(
            "%s has not implemented the solve method"
            % self.__class__
        )
