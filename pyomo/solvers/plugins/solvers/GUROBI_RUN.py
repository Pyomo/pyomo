#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import re


"""This script is run using the Gurobi/system python. Do not assume any
third party packages are available!

"""
from gurobipy import gurobi, read, GRB
import sys

if sys.version_info[0] < 3:
    from itertools import izip as zip

GUROBI_VERSION = gurobi.version()

# NOTE: this function / module is independent of Pyomo, and only relies on the
#       GUROBI python bindings. consequently, nothing in this function should
#       throw an exception that is expected to be handled by Pyomo - it won't be.
#       rather, print an error message and return - the caller will know to look
#       in the logs in case of a failure.


def _is_numeric(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


def gurobi_run(model_file, warmstart_file, mipgap, options, suffixes):
    # figure out what suffixes we need to extract.
    extract_duals = False
    extract_slacks = False
    extract_reduced_costs = False
    for suffix in suffixes:
        flag = False
        if re.match(suffix, "dual"):
            extract_duals = True
            flag = True
        if re.match(suffix, "slack"):
            extract_slacks = True
            flag = True
        if re.match(suffix, "rc"):
            extract_reduced_costs = True
            flag = True
        if not flag:
            print(
                "***The GUROBI solver plugin cannot extract solution suffix=" + suffix
            )
            return

    # Load the lp model
    model = read(model_file)

    # if the use wants to extract duals or reduced costs and the
    # model has quadratic constraints then we need to set the
    # QCPDual param to 1 (which apparently makes the solve more
    # expensive in the quadratic case). If we do not set this param
    # and and we attempt to access these suffixes in the solution
    # printing the module will crash (when we have a QCP)
    if GUROBI_VERSION[0] >= 5:
        if (extract_reduced_costs is True) or (extract_duals is True):
            model.setParam(GRB.Param.QCPDual, 1)

    if model is None:
        print(
            "***The GUROBI solver plugin failed to load the input LP file=" + model_file
        )
        return

    if warmstart_file is not None:
        model.read(warmstart_file)

    # set the mipgap if specified.
    if mipgap is not None:
        model.setParam("MIPGap", mipgap)

    # set all other solver parameters, if specified.
    # GUROBI doesn't throw an exception if an unknown
    # key is specified, so you have to stare at the
    # output to see if it was accepted.
    for key, value in options.items():
        # When options come from the pyomo command, all
        # values are string types, so we try to cast
        # them to a numeric value in the event that
        # setting the parameter fails.
        try:
            model.setParam(key, value)
        except TypeError:
            # we place the exception handling for checking
            # the cast of value to a float in another
            # function so that we can simply call raise here
            # instead of except TypeError as e / raise e,
            # because the latter does not preserve the
            # Gurobi stack trace
            if not _is_numeric(value):
                model.close()
                raise
            model.setParam(key, float(value))

    if 'relax_integrality' in options:
        for v in model.getVars():
            if v.vType != GRB.CONTINUOUS:
                v.vType = GRB.CONTINUOUS
        model.update()

    # optimize the model
    model.optimize()

    # This attribute needs to be extracted before
    # calling model.getVars() or model.getConstrs().
    # Apparently it gets reset by those methods.
    wall_time = model.getAttr(GRB.Attr.Runtime)

    solver_status = model.getAttr(GRB.Attr.Status)
    solution_status = None
    return_code = 0
    if solver_status == GRB.LOADED:
        status = 'aborted'
        message = 'Model is loaded, but no solution information is available.'
        term_cond = 'error'
        solution_status = 'unknown'
    elif solver_status == GRB.OPTIMAL:
        status = 'ok'
        message = 'Model was solved to optimality (subject to tolerances), and an optimal solution is available.'
        term_cond = 'optimal'
        solution_status = 'optimal'
    elif solver_status == GRB.INFEASIBLE:
        status = 'warning'
        message = 'Model was proven to be infeasible.'
        term_cond = 'infeasible'
        solution_status = 'infeasible'
    elif solver_status == GRB.INF_OR_UNBD:
        status = 'warning'
        message = 'Problem proven to be infeasible or unbounded.'
        term_cond = 'infeasibleOrUnbounded'
        solution_status = 'unsure'
    elif solver_status == GRB.UNBOUNDED:
        status = 'warning'
        message = 'Model was proven to be unbounded.'
        term_cond = 'unbounded'
        solution_status = 'unbounded'
    elif solver_status == GRB.CUTOFF:
        status = 'aborted'
        message = 'Optimal objective for model was proven to be worse than the value specified in the Cutoff  parameter. No solution information is available.'
        term_cond = 'minFunctionValue'
        solution_status = 'unknown'
    elif solver_status == GRB.ITERATION_LIMIT:
        status = 'aborted'
        message = 'Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterationLimit parameter.'
        term_cond = 'maxIterations'
        solution_status = 'stoppedByLimit'
    elif solver_status == GRB.NODE_LIMIT:
        status = 'aborted'
        message = 'Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter.'
        term_cond = 'maxEvaluations'
        solution_status = 'stoppedByLimit'
    elif solver_status == GRB.TIME_LIMIT:
        status = 'aborted'
        message = 'Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter.'
        term_cond = 'maxTimeLimit'
        solution_status = 'stoppedByLimit'
    elif hasattr(GRB, "WORK_LIMIT") and (solver_status == GRB.WORK_LIMIT):
        status = 'aborted'
        message = 'Optimization terminated because the work expended exceeded the value specified in the WorkLimit parameter.'
        term_cond = 'maxTimeLimit'
        solution_status = 'stoppedByLimit'
    elif solver_status == GRB.SOLUTION_LIMIT:
        status = 'aborted'
        message = 'Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter.'
        term_cond = 'stoppedByLimit'
        solution_status = 'stoppedByLimit'
    elif solver_status == GRB.INTERRUPTED:
        status = 'aborted'
        message = 'Optimization was terminated by the user.'
        term_cond = 'error'
        solution_status = 'error'
    elif solver_status == GRB.NUMERIC:
        status = 'error'
        message = (
            'Optimization was terminated due to unrecoverable numerical difficulties.'
        )
        term_cond = 'error'
        solution_status = 'error'
    elif solver_status == GRB.SUBOPTIMAL:
        status = 'warning'
        message = 'Unable to satisfy optimality tolerances; a sub-optimal solution is available.'
        term_cond = 'other'
        solution_status = 'feasible'
    # note that USER_OBJ_LIMIT was added in Gurobi 7.0, so it may not be present
    elif (solver_status is not None) and (
        solver_status == getattr(GRB, 'USER_OBJ_LIMIT', None)
    ):
        status = 'aborted'
        message = (
            "User specified an objective limit "
            "(a bound on either the best objective "
            "or the best bound), and that limit has "
            "been reached. Solution is available."
        )
        term_cond = 'other'
        solution_status = 'stoppedByLimit'
    else:
        status = 'error'
        message = "Unhandled Gurobi solve status (" + str(solver_status) + ")"
        term_cond = 'error'
        solution_status = 'error'
    assert solution_status is not None

    sense = model.getAttr(GRB.Attr.ModelSense)
    try:
        obj_value = model.getAttr(GRB.Attr.ObjVal)
    except:
        obj_value = None
        if term_cond == "unbounded":
            if sense < 0:
                # maximize
                obj_value = float('inf')
            else:
                # minimize
                obj_value = float('-inf')
        elif term_cond == "infeasible":
            if sense < 0:
                # maximize
                obj_value = float('-inf')
            else:
                # minimize
                obj_value = float('inf')

    result = {}
    problem = result['problem'] = {}
    problem['name'] = model.getAttr(GRB.Attr.ModelName)

    # TODO: find out about bounds and fix this with error checking
    # this line fails for some reason so set the value to unknown
    try:
        bound = model.getAttr(GRB.Attr.ObjBound)
    except Exception:
        if term_cond == 'optimal':
            bound = obj_value
        else:
            bound = None

    if sense < 0:
        problem["sense"] = "maximize"
        if bound is None:
            bound = float('inf')
        problem["upper_bound"] = bound
    else:
        problem["sense"] = "minimize"
        if bound is None:
            bound = float('-inf')
        problem["lower_bound"] = bound

    # TODO: Get the number of objective functions from GUROBI
    n_objs = 1
    problem["number_of_objectives"] = n_objs

    cons = model.getConstrs()
    qcons = []
    if GUROBI_VERSION[0] >= 5:
        qcons = model.getQConstrs()
    problem["number_of_constraints"] = len(cons) + len(qcons) + model.NumSOS

    vars = model.getVars()
    problem["number_of_variables"] = len(vars)

    n_binvars = model.getAttr(GRB.Attr.NumBinVars)
    problem["number_of_binary_variables"] = n_binvars

    n_intvars = model.getAttr(GRB.Attr.NumIntVars)
    problem["number_of_integer_variables"] = n_intvars
    problem["number_of_continuous_variables"] = len(vars) - n_intvars
    problem["number_of_nonzeros"] = model.getAttr(GRB.Attr.NumNZs)

    # write out the information required by results.solver
    solver = result['solver'] = {}

    solver['status'] = status
    solver['return_code'] = return_code
    solver['message'] = message
    solver['wall_time'] = wall_time
    solver['termination_condition'] = term_cond
    solver['termination_message'] = message

    is_discrete = False
    if model.getAttr(GRB.Attr.IsMIP):
        is_discrete = True

    if (term_cond == 'optimal') or (model.getAttr(GRB.Attr.SolCount) >= 1):
        solution = result['solution'] = {}
        solution['status'] = solution_status
        solution['message'] = message
        solution['objective'] = obj_value
        solution['gap'] = 0.0

        vals = model.getAttr("X", vars)
        names = model.getAttr("VarName", vars)
        solution['var'] = {name: val for name, val in zip(names, vals)}

        if extract_reduced_costs and not is_discrete:
            vals = model.getAttr("Rc", vars)
            solution['varrc'] = {name: val for name, val in zip(names, vals)}

        if extract_duals or extract_slacks:
            con_names = model.getAttr("ConstrName", cons)
            if GUROBI_VERSION[0] >= 5:
                qcon_names = model.getAttr("QCName", qcons)

        if extract_duals and not is_discrete:
            # Pi attributes in Gurobi are the constraint duals
            vals = model.getAttr("Pi", cons)
            solution['constraintdual'] = {
                name: val for name, val in zip(con_names, vals)
            }
            if GUROBI_VERSION[0] >= 5:
                # QCPI attributes in Gurobi are the constraint duals
                vals = model.getAttr("QCPi", qcons)
                solution['constraintdual'].update(zip(qcon_names, vals))

        if extract_slacks:
            vals = model.getAttr("Slack", cons)
            solution['constraintslack'] = {
                name: val for name, val in zip(con_names, vals)
            }
            if GUROBI_VERSION[0] >= 5:
                vals = model.getAttr("QCSlack", qcons)
                solution['constraintslack'].update(zip(qcon_names, vals))

    model.close()
    model = None
    return result


def write_result(result, soln_file):
    with open(soln_file, "w+") as FILE:
        for section, data in result.items():
            FILE.write(f'section:{section}\n')
            for key, val in data.items():
                if val.__class__ is dict:
                    for name, v in val.items():
                        FILE.write(f'{key}:{name}:{v}\n')
                else:
                    FILE.write(f'{key}:{val}\n')
