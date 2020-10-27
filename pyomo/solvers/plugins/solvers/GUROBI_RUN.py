#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import re


"""
This script is run using the Gurobi/system python. Do not assume any third party packages
are available!
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

def gurobi_run(model_file, warmstart_file, soln_file, mipgap, options, suffixes):

    # figure out what suffixes we need to extract.
    extract_duals = False
    extract_slacks = False
    extract_reduced_costs = False
    for suffix in suffixes:
        flag=False
        if re.match(suffix,"dual"):
            extract_duals = True
            flag=True
        if re.match(suffix,"slack"):
            extract_slacks = True
            flag=True
        if re.match(suffix,"rc"):
            extract_reduced_costs = True
            flag=True
        if not flag:
            print("***The GUROBI solver plugin cannot extract solution suffix="+suffix)
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
            model.setParam(GRB.Param.QCPDual,1)

    if model is None:
        print("***The GUROBI solver plugin failed to load the input LP file="+soln_file)
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
    if (solver_status == GRB.LOADED):
        status = 'aborted'
        message = 'Model is loaded, but no solution information is availale.'
        term_cond = 'error'
        solution_status = 'unknown'
    elif (solver_status == GRB.OPTIMAL):
        status = 'ok'
        message = 'Model was solved to optimality (subject to tolerances), and an optimal solution is available.'
        term_cond = 'optimal'
        solution_status = 'optimal'
    elif (solver_status == GRB.INFEASIBLE):
        status = 'warning'
        message = 'Model was proven to be infeasible.'
        term_cond = 'infeasible'
        solution_status = 'infeasible'
    elif (solver_status == GRB.INF_OR_UNBD):
        status = 'warning'
        message = 'Problem proven to be infeasible or unbounded.'
        term_cond = 'infeasibleOrUnbounded'
        solution_status = 'unsure'
    elif (solver_status == GRB.UNBOUNDED):
        status = 'warning'
        message = 'Model was proven to be unbounded.'
        term_cond = 'unbounded'
        solution_status = 'unbounded'
    elif (solver_status == GRB.CUTOFF):
        status = 'aborted'
        message = 'Optimal objective for model was proven to be worse than the value specified in the Cutoff  parameter. No solution information is available.'
        term_cond = 'minFunctionValue'
        solution_status = 'unknown'
    elif (solver_status == GRB.ITERATION_LIMIT):
        status = 'aborted'
        message = 'Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterationLimit parameter.'
        term_cond = 'maxIterations'
        solution_status = 'stoppedByLimit'
    elif (solver_status == GRB.NODE_LIMIT):
        status = 'aborted'
        message = 'Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter.'
        term_cond = 'maxEvaluations'
        solution_status = 'stoppedByLimit'
    elif (solver_status == GRB.TIME_LIMIT):
        status = 'aborted'
        message = 'Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter.'
        term_cond = 'maxTimeLimit'
        solution_status = 'stoppedByLimit'
    elif (solver_status == GRB.SOLUTION_LIMIT):
        status = 'aborted'
        message = 'Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter.'
        term_cond = 'stoppedByLimit'
        solution_status = 'stoppedByLimit'
    elif (solver_status == GRB.INTERRUPTED):
        status = 'aborted'
        message = 'Optimization was terminated by the user.'
        term_cond = 'error'
        solution_status = 'error'
    elif (solver_status == GRB.NUMERIC):
        status = 'error'
        message = 'Optimization was terminated due to unrecoverable numerical difficulties.'
        term_cond = 'error'
        solution_status = 'error'
    elif (solver_status == GRB.SUBOPTIMAL):
        status = 'warning'
        message = 'Unable to satisfy optimality tolerances; a sub-optimal solution is available.'
        term_cond = 'other'
        solution_status = 'feasible'
    # note that USER_OBJ_LIMIT was added in Gurobi 7.0, so it may not be present
    elif (solver_status is not None) and \
         (solver_status == getattr(GRB,'USER_OBJ_LIMIT',None)):
        status = 'aborted'
        message = "User specified an objective limit " \
                  "(a bound on either the best objective " \
                  "or the best bound), and that limit has " \
                  "been reached. Solution is available."
        term_cond = 'other'
        solution_status = 'stoppedByLimit'
    else:
        status = 'error'
        message = ("Unhandled Gurobi solve status "
                   "("+str(solver_status)+")")
        term_cond = 'error'
        solution_status = 'error'
    assert solution_status is not None

    sense = model.getAttr(GRB.Attr.ModelSense)
    try:
        obj_value = model.getAttr(GRB.Attr.ObjVal)
    except:
        obj_value = None
        if term_cond == "unbounded":
            if (sense < 0):
                # maximize
                obj_value = float('inf')
            else:
                # minimize
                obj_value = float('-inf')
        elif term_cond == "infeasible":
            if (sense < 0):
                # maximize
                obj_value = float('-inf')
            else:
                # minimize
                obj_value = float('inf')

    # write the solution file
    solnfile = open(soln_file, "w+")

    # write the information required by results.problem
    solnfile.write("section:problem\n")
    name = model.getAttr(GRB.Attr.ModelName)
    solnfile.write("name: "+name+'\n')

    # TODO: find out about bounds and fix this with error checking
    # this line fails for some reason so set the value to unknown
    try:
        bound = model.getAttr(GRB.Attr.ObjBound)
    except Exception:
        if term_cond == 'optimal':
            bound = obj_value
        else:
            bound = None

    if (sense < 0):
        solnfile.write("sense:maximize\n")
        if bound is None:
            solnfile.write("upper_bound: %f\n" % float('inf'))
        else:
            solnfile.write("upper_bound: %s\n" % str(bound))
    else:
        solnfile.write("sense:minimize\n")
        if bound is None:
            solnfile.write("lower_bound: %f\n" % float('-inf'))
        else:
            solnfile.write("lower_bound: %s\n" % str(bound))

    # TODO: Get the number of objective functions from GUROBI
    n_objs = 1
    solnfile.write("number_of_objectives: %d\n" % n_objs)

    cons = model.getConstrs()
    qcons = []
    if GUROBI_VERSION[0] >= 5:
        qcons = model.getQConstrs()
    solnfile.write("number_of_constraints: %d\n" % (len(cons)+len(qcons)+model.NumSOS,))

    vars = model.getVars()
    solnfile.write("number_of_variables: %d\n" % len(vars))

    n_binvars = model.getAttr(GRB.Attr.NumBinVars)
    solnfile.write("number_of_binary_variables: %d\n" % n_binvars)

    n_intvars = model.getAttr(GRB.Attr.NumIntVars)
    solnfile.write("number_of_integer_variables: %d\n" % n_intvars)

    solnfile.write("number_of_continuous_variables: %d\n" % (len(vars)-n_intvars,))

    solnfile.write("number_of_nonzeros: %d\n" % model.getAttr(GRB.Attr.NumNZs))

    # write out the information required by results.solver
    solnfile.write("section:solver\n")

    solnfile.write('status: %s\n' % status)
    solnfile.write('return_code: %s\n' % return_code)
    solnfile.write('message: %s\n' % message)
    solnfile.write('wall_time: %s\n' % str(wall_time))
    solnfile.write('termination_condition: %s\n' % term_cond)
    solnfile.write('termination_message: %s\n' % message)

    is_discrete = False
    if (model.getAttr(GRB.Attr.IsMIP)):
        is_discrete = True

    if (term_cond == 'optimal') or (model.getAttr(GRB.Attr.SolCount) >= 1):
        solnfile.write('section:solution\n')
        solnfile.write('status: %s\n' % (solution_status))
        solnfile.write('message: %s\n' % message)
        solnfile.write('objective: %s\n' % str(obj_value))
        solnfile.write('gap: 0.0\n')

        vals = model.getAttr("X", vars)
        names = model.getAttr("VarName", vars)
        for val, name in zip(vals, names):
            solnfile.write('var: %s : %s\n' % (str(name), str(val)))

        if (is_discrete is False) and (extract_reduced_costs is True):
            vals = model.getAttr("Rc", vars)
            for val, name in zip(vals, names):
                solnfile.write('varrc: %s : %s\n' % (str(name), str(val)))

        if extract_duals or extract_slacks:
            con_names = model.getAttr("ConstrName", cons)
            if GUROBI_VERSION[0] >= 5:
                qcon_names = model.getAttr("QCName", qcons)

        if (is_discrete is False) and (extract_duals is True):
            vals = model.getAttr("Pi", cons)
            for val, name in zip(vals, con_names):
               # Pi attributes in Gurobi are the constraint duals
                solnfile.write("constraintdual: %s : %s\n" % (str(name), str(val)))
            if GUROBI_VERSION[0] >= 5:
                vals = model.getAttr("QCPi", qcons)
                for val, name in zip(vals, qcon_names):
                    # QCPI attributes in Gurobi are the constraint duals
                    solnfile.write("constraintdual: %s : %s\n" % (str(name), str(val)))

        if (extract_slacks is True):
            vals = model.getAttr("Slack", cons)
            for val, name in zip(vals, con_names):
                solnfile.write("constraintslack: %s : %s\n" % (str(name), str(val)))
            if GUROBI_VERSION[0] >= 5:
                vals = model.getAttr("QCSlack", qcons)
                for val, name in zip(vals, qcon_names):
                    solnfile.write("constraintslack: %s : %s\n" % (str(name), str(val)))

    solnfile.close()
