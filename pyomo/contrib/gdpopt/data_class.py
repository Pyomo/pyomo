"""Module for GDPopt data classes."""


class GDPoptSolveData(object):
    """Data container to hold solve-instance data.

    Key attributes:
        - original_model: the original model that the user gave us to solve
        - working_model: the original model after preprocessing
        - linear_GDP: the linear-discrete master problem
        - results: Pyomo results objective
        - timing: dictionary of time elapsed for solver functions

    """
    pass


class MasterProblemResult(object):
    """Data class for master problem results data.

    Key attributes:
        - feasible: True/False if feasible solution obtained
        - var_values: list of variable values
        - pyomo_results: results object from solve() statement
        - disjunct_values: list of disjunct values

    """
    pass


class SubproblemResult(object):
    """Data class for subproblem results data.

    Key attributes:
        - feasible: True/False if feasible solution obtained
        - var_values: list of variable values
        - dual_values: list of constraint dual values
        - pyomo_results: results object from solve() statement

    """
