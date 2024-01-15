"""
This module contains functions for computing an irreducible infeasible set
for a Pyomo MILP or LP using a specified commercial solver, one of CPLEX,
Gurobi, or Xpress.
"""

import abc
import logging
from pyomo.environ import SolverFactory

logger = logging.getLogger("pyomo.contrib.iis")
logger.setLevel(logging.INFO)


def write_iis(pyomo_model, iis_file_name, solver=None, logger=logger):
    """
    Write an irreducible infeasible set for a Pyomo MILP or LP
    using the specified commercial solver.

    Arguments
    ---------
        pyomo_model:
            A Pyomo Block or ConcreteModel
        iis_file_name:str
            A file name to write the IIS to, e.g., infeasible_model.ilp
        solver:str
            Specify the solver to use, one of "cplex", "gurobi", or "xpress".
            If None, the tool will use the first solver available.
        logger:logging.Logger
            A logger for messages. Uses pyomo.contrib.iis logger by default.

    Returns
    -------
        iis_file_name:str
            The file containing the IIS.
    """

    available_solvers = [
        s
        for s, sp in zip(_supported_solvers, _supported_solvers_persistent)
        if SolverFactory(sp).available(exception_flag=False)
    ]

    if solver is None:
        if len(available_solvers) == 0:
            raise RuntimeError(
                f"Could not find a solver to use, supported solvers are {_supported_solvers}"
            )
        solver = available_solvers[0]
        logger.info(f"Using solver {solver}")
    else:
        # validate
        solver = solver.lower()
        solver = _remove_suffix(solver, "_persistent")
        if solver not in available_solvers:
            raise RuntimeError(
                f"The Pyomo persistent interface to {solver} could not be found."
            )

    solver_name = solver
    solver = SolverFactory(solver + "_persistent")

    solver.set_instance(pyomo_model, symbolic_solver_labels=True)

    iis = IISFactory(solver)
    iis.compute()
    iis_file_name = iis.write(iis_file_name)
    logger.info(f"IIS written to {iis_file_name}")
    return iis_file_name


def _remove_suffix(string, suffix):
    if string.endswith(suffix):
        return string[: -len(suffix)]
    else:
        return string


class _IISBase(abc.ABC):
    def __init__(self, solver):
        self._solver = solver

    @abc.abstractmethod
    def compute(self):
        """computes the IIS/Conflict"""
        pass

    @abc.abstractmethod
    def write(self, file_name):
        """writes the IIS in LP format
        return the file name written
        """
        pass


class CplexConflict(_IISBase):
    def compute(self):
        self._solver._solver_model.conflict.refine()

    def write(self, file_name):
        self._solver._solver_model.conflict.write(file_name)
        return file_name


class GurobiIIS(_IISBase):
    def compute(self):
        self._solver._solver_model.computeIIS()

    def write(self, file_name):
        # gurobi relies on the suffix to
        # determine the file type
        file_name = _remove_suffix(file_name, ".ilp")
        file_name += ".ilp"
        self._solver._solver_model.write(file_name)
        return file_name


class XpressIIS(_IISBase):
    def compute(self):
        self._solver._solver_model.iisfirst(1)

    def write(self, file_name):
        self._solver._solver_model.iiswrite(0, file_name, 0, "l")
        if self._solver._version[0] < 38:
            return file_name
        else:
            return _remove_suffix(file_name, ".lp") + ".lp"


_solver_map = {
    "cplex_persistent": CplexConflict,
    "gurobi_persistent": GurobiIIS,
    "xpress_persistent": XpressIIS,
}


def IISFactory(solver):
    if solver.name not in _solver_map:
        raise RuntimeError(f"Unrecognized solver {solver.name}")
    return _solver_map[solver.name](solver)


_supported_solvers_persistent = list(_solver_map.keys())
_supported_solvers = [_remove_suffix(s, "_persistent") for s in _solver_map]
