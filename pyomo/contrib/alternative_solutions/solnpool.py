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

import pyomo.environ as pe
from pyomo.contrib import appsi
from pyomo.contrib.alternative_solutions import aos_utils, solution
import gurobipy
import pdb


def gurobi_generate_solutions(
    model,
    num_solutions=10,
    rel_opt_gap=None,
    abs_opt_gap=None,
    solver_options={},
    tee=False,
    quiet=True,
):
    """
    Finds alternative optimal solutions for discrete variables using Gurobi's
    built-in Solution Pool capability. See the Gurobi Solution Pool
    documentation for additional details.

        Parameters
        ----------
        model : ConcreteModel
            A concrete Pyomo model.
        num_solutions : int
            The maximum number of solutions to generate. This parameter maps to
            the PoolSolutions parameter in Gurobi.
        rel_opt_gap : non-negative float or None
            The relative optimality gap for allowable alternative solutions.
            None implies that there is no limit on the relative optimality gap
            (i.e. that any feasible solution can be considered by Gurobi).
            This parameter maps to the PoolGap parameter in Gurobi.
        abs_opt_gap : non-negative float or None
            The absolute optimality gap for allowable alternative solutions.
            None implies that there is no limit on the absolute optimality gap
            (i.e. that any feasible solution can be considered by Gurobi).
            This parameter maps to the PoolGapAbs parameter in Gurobi.
        solver_options : dict
            Solver option-value pairs to be passed to the Gurobi solver.
        tee : boolean
            Boolean indicating that the solver output should be displayed.

        Returns
        -------
        solutions
            A list of Solution objects.  [Solution]
    """
    #
    # Setup gurobi
    #
    opt = appsi.solvers.Gurobi()
    if not opt.available():     #pragma: no cover
        return []

    opt.config.stream_solver = tee
    opt.config.load_solution = False
    opt.gurobi_options["PoolSolutions"] = num_solutions
    opt.gurobi_options["PoolSearchMode"] = 2
    if rel_opt_gap is not None:
        opt.gurobi_options["PoolGap"] = rel_opt_gap
    if abs_opt_gap is not None:
        opt.gurobi_options["PoolGapAbs"] = abs_opt_gap
    for parameter, value in solver_options.items():
        opt.gurobi_options[parameter] = value
    #
    # Run gurobi
    #
    results = opt.solve(model)
    condition = results.termination_condition
    if not (condition == appsi.base.TerminationCondition.optimal):
        if not quiet:
            print(
                (
                    "Model cannot be solved, "
                    "TerminationCondition = {}"
                ).format(condition.value)
            )
        return []
    #
    # Collect solutions
    #
    solution_count = opt.get_model_attr("SolCount")
    variables = aos_utils.get_model_variables(model, "all", include_fixed=True)
    solutions = []
    for i in range(solution_count):
        #
        # Load the i-th solution into the model
        #
        results.solution_loader.load_vars(solution_number=i)
        #
        # Pull the solution from the model into a Solution object,
        # and append to our list of solutions
        #
        solutions.append(solution.Solution(model, variables))

    return solutions
