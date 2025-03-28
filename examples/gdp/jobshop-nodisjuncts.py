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

import pyomo.environ as pyo
from pyomo.gdp import Disjunction

#
# Jobshop example from http://www.gams.com/modlib/libhtml/logmip4.htm
#
# This model solves a jobshop scheduling, which has a set of jobs
# which must be processed in sequence of stages but not all jobs
# require all stages. A zero wait transfer policy is assumed between
# stages. To obtain a feasible solution it is necessary to eliminate
# all clashes between jobs. It requires that no two jobs be performed
# at any stage at any time. The objective is to minimize the makespan,
# the time to complete all jobs.
#
# References:
#
# Raman & Grossmann, Computers and Chemical Engineering 18, 7, p.563-578, 1994.
#
# Aldo Vecchietti, LogMIP User's Manual, http://www.logmip.ceride.gov.ar/, 2007
#


def build_model():
    model = pyo.AbstractModel()

    model.JOBS = pyo.Set(ordered=True)
    model.STAGES = pyo.Set(ordered=True)
    model.I_BEFORE_K = pyo.RangeSet(0, 1)

    # Task durations
    model.tau = pyo.Param(model.JOBS, model.STAGES, default=0)

    # Total Makespan (this will be the objective)
    model.ms = pyo.Var()

    # Start time of each job
    def t_bounds(model, I):
        return (0, sum(pyo.value(model.tau[idx]) for idx in model.tau))

    model.t = pyo.Var(model.JOBS, within=pyo.NonNegativeReals, bounds=t_bounds)

    # Auto-generate the L set (potential collisions between 2 jobs at any stage.
    def _L_filter(model, I, K, J):
        return I < K and model.tau[I, J] and model.tau[K, J]

    model.L = pyo.Set(
        initialize=model.JOBS * model.JOBS * model.STAGES, dimen=3, filter=_L_filter
    )

    # Makespan is greater than the start time of every job + that job's
    # total duration
    def _feas(model, I):
        return model.ms >= model.t[I] + sum(model.tau[I, M] for M in model.STAGES)

    model.Feas = pyo.Constraint(model.JOBS, rule=_feas)

    # Define the disjunctions: either job I occurs before K or K before I
    def _disj(model, I, K, J):
        lhs = model.t[I] + sum([M < J and model.tau[I, M] or 0 for M in model.STAGES])
        rhs = model.t[K] + sum([M < J and model.tau[K, M] or 0 for M in model.STAGES])
        return [lhs + model.tau[I, J] <= rhs, rhs + model.tau[K, J] <= lhs]

    model.disj = Disjunction(model.L, rule=_disj)

    # minimize makespan
    model.makespan = pyo.Objective(expr=model.ms)
    return model
