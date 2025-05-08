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

import logging
import os
import re
import sys
import time

from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.numvalue import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
    DirectOrPersistentSolver,
)
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory
from pyomo.core.base.suffix import Suffix
import pyomo.core.base.var


logger = logging.getLogger('pyomo.solvers')


class DegreeError(ValueError):
    pass


def _is_convertible(conv_type, x):
    try:
        conv_type(x)
    except ValueError:
        return False
    return True


def _print_message(xp_prob, _, msg, *args):
    if msg is not None:
        sys.stdout.write(msg + '\n')
        sys.stdout.flush()


def _finalize_xpress_import(xpress, avail):
    if not avail:
        return
    XpressDirect._version = tuple(int(k) for k in xpress.getversion().split('.'))
    XpressDirect._name = "Xpress %s.%s.%s" % XpressDirect._version
    # in versions prior to 34, xpress raised a RuntimeError, but
    # in more recent versions it raises a
    # xpress.ModelError. We'll cache the appropriate one here
    if XpressDirect._version[0] < 34:
        XpressDirect.XpressException = RuntimeError
    else:
        XpressDirect.XpressException = xpress.ModelError
    # In (pypi) versions prior to 8.13.0, the 'xpress.rng' keyword was
    # 'xpress.range'
    if not hasattr(xpress, 'rng'):
        xpress.rng = xpress.range

    #
    # Xpress 9.5 (44.1.1) changed the Python API fairly significantly.
    # We will map between the two APIs based on the version.
    #
    if XpressDirect._version < (44,):

        def _addConstraint(
            self,
            prob,
            constraint=None,
            body=None,
            lb=None,
            ub=None,
            type=None,
            rhs=None,
            name='',
        ):
            # It's unclear what the acceptable "default" values are for
            # lb, ub, etc. (putting in the values from the documentation
            # generates errors).  We will instead use None and filter
            # out any non-None fields.
            args = {'sense': type, 'name': name}
            for field in ('constraint', 'body', 'lb', 'ub', 'rhs'):
                if locals()[field] is not None:
                    args[field] = locals()[field]
            con = xpress.constraint(**args)
            prob.addConstraint(con)
            return con

        def _addVariable(self, prob, name, lb, ub, vartype):
            var = xpress.var(name=name, lb=lb, ub=ub, vartype=vartype)
            prob.addVariable(var)
            return var

        def _addSOS(self, prob, indices, weights, type, name):
            con = xpress.sos(indices, weights, type, name)
            prob.addSOS(con)
            return con

        XpressDirect._addConstraint = _addConstraint
        XpressDirect._addVariable = _addVariable
        XpressDirect._addSOS = _addSOS
        XpressDirect._getSlacks = lambda self, prob, con: prob.getSlack(con)
        XpressDirect._getDuals = lambda self, prob, con: prob.getDual(con)
        XpressDirect._getRedCosts = lambda self, prob, con: prob.getRCost(con)
    else:
        # Note that rhsrange (the last argument) was not added until
        # 9.5.  We will not include it here in the compatibility
        # wrapper.
        def _addConstraint(
            self,
            prob,
            constraint=None,
            body=None,
            lb=None,
            ub=None,
            type=None,
            rhs=None,
            name='',
        ):
            con = xpress.constraint(
                constraint=constraint,
                body=body,
                lb=lb,
                ub=ub,
                type=type,
                rhs=rhs,
                name=name,
            )
            prob.addConstraint(con)
            return con

        XpressDirect._addConstraint = _addConstraint
        XpressDirect._addVariable = (
            lambda self, prob, name, lb, ub, vartype: prob.addVariable(
                name=name, lb=lb, ub=ub, vartype=vartype
            )
        )
        XpressDirect._addSOS = (
            lambda self, prob, indices, weights, type, name: prob.addSOS(
                indices, weights, type, name
            )
        )
        XpressDirect._getSlacks = lambda self, prob, con: prob.getSlacks(con)
        XpressDirect._getDuals = lambda self, prob, con: prob.getDuals(con)
        XpressDirect._getRedCosts = lambda self, prob, con: prob.getRedCosts(con)

        # Note that as of 9.5, xpress.var raises an exception when
        # compared using '==' after it has been removed from the model.
        # This can foul up ComponentMaps in the persistent interface,
        # so we will hard-code the `var` as not being hashable (so the
        # ComponentMap will use the id() as the key)
        ComponentMap.hasher.hashable(xpress.var, False)


class _xpress_importer_class(object):
    # We want to be able to *update* the message that the deferred
    # import generates using the stdout recorded during the actual
    # import.  As strings are immutable in Python, we will give this
    # *class* as the error message, so that we can update the embedded
    # string later.
    def __init__(self):
        self.import_message = ""

    def __str__(self):
        return str(self.import_message)

    def __call__(self):
        _cwd = os.getcwd()
        try:
            with capture_output() as OUT:
                import xpress
        finally:
            # In some versions of XPRESS (notably 8.9.0), `import
            # xpress` temporarily changes the CWD.  If the import fails
            # (e.g., due to an expired license), the CWD is not always
            # restored.  This block ensures that the CWD is preserved.
            os.chdir(_cwd)
            self.import_message += OUT.getvalue()
        return xpress


@SolverFactory.register('xpress_direct', doc='Direct python interface to XPRESS')
class XpressDirect(DirectSolver):
    _name = None
    _version = None
    XpressException = RuntimeError

    def __init__(self, **kwds):
        if 'type' not in kwds:
            kwds['type'] = 'xpress_direct'
        super(XpressDirect, self).__init__(**kwds)
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()

        self._range_constraints = set()

        self._python_api_exists = xpress_available

        # TODO: this isn't a limit of XPRESS, which implements an SLP
        #       method for NLPs. But it is a limit of *this* interface
        self._max_obj_degree = 2
        self._max_constraint_degree = 2

        # There does not seem to be an easy way to get the
        # wallclock time out of xpress, so we will measure it
        # ourselves
        self._opt_time = None

        # Note: Undefined capabilities default to None
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

        # remove the instance-level definition of the xpress version:
        # because the version comes from an imported module, only one
        # version of xpress is supported (and stored as a class attribute)
        del self._version

        # xpress will apply the warmstart itself instead of
        # DirectOrPersistentSolver._presolve
        self._apply_warmstart = False

    def available(self, exception_flag=True):
        """True if the solver is available."""

        if not xpress_available:
            if exception_flag:
                xpress.log_import_warning(logger=__name__)
                raise ApplicationError(
                    "No Python bindings available for %s solver plugin" % (type(self),)
                )
            return False

        # Check that there is a valid license
        try:
            xpress.init()
            return True
        except:
            if exception_flag:
                raise
            return False
        finally:
            xpress.free()

    def _presolve(self, *args, **kwds):
        # we'll apply the warmstart in _solve_model so the
        # message "User solution (_) stored" is wrote to the
        # correct place, i.e., the console or the log or both
        self._apply_warmstart = kwds.pop("warmstart", False)
        return super()._presolve(*args, **kwds)

    def _apply_solver(self):
        StaleFlagManager.mark_all_as_stale()

        self._solver_model.setlogfile(self._log_file)
        if self._keepfiles:
            print("Solver log file: " + self._log_file)

        # Setting a log file in xpress disables all output
        # in xpress versions less than 36.
        # This callback prints all messages to stdout
        # when using those xpress versions.
        if self._tee and XpressDirect._version[0] < 36:
            self._solver_model.addcbmessage(_print_message, None, 0)

        # set xpress options
        # if the user specifies a 'mipgap', set it, and
        # set xpress's related options to 0.
        if self.options.mipgap is not None:
            self._solver_model.setControl('miprelstop', float(self.options.mipgap))
            self._solver_model.setControl('miprelcutoff', 0.0)
            self._solver_model.setControl('mipaddcutoff', 0.0)
        # xpress is picky about the type which is passed
        # into a control. So we will infer and cast
        # get the xpress valid controls
        xp_controls = xpress.controls
        for key, option in self.options.items():
            if key == 'mipgap':  # handled above
                continue
            try:
                self._solver_model.setControl(key, option)
            except XpressDirect.XpressException:
                # take another try, converting to its type
                # we'll wrap this in a function to raise the
                # xpress error
                contr_type = type(getattr(xp_controls, key))
                if not _is_convertible(contr_type, option):
                    raise
                self._solver_model.setControl(key, contr_type(option))

        start_time = time.time()
        if self._tee:
            self._solve_model()
        else:
            # In xpress versions greater than or equal 36,
            # it seems difficult to completely suppress console
            # output without disabling logging altogether.
            # As a work around, we capature all screen output
            # when tee is False.
            with capture_output() as OUT:
                self._solve_model()
        self._opt_time = time.time() - start_time

        self._solver_model.setlogfile('')
        if self._tee and XpressDirect._version[0] < 36:
            self._solver_model.removecbmessage(_print_message, None)

        # FIXME: can we get a return code indicating if XPRESS had a
        # significant failure?
        return Bunch(rc=None, log=None)

    def _get_mip_results(self, results, soln):
        """Sets up `results` and `soln` and returns whether there is a solution
        to query.
        Returns `True` if a feasible solution is available, `False` otherwise.
        """
        xprob = self._solver_model
        xp = xpress
        xprob_attrs = xprob.attributes
        status = xprob_attrs.mipstatus
        mip_sols = xprob_attrs.mipsols
        if status == xp.mip_not_loaded:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_message = (
                "Model is not loaded; no solution information is available."
            )
            results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.unknown
            # no MIP solution, first LP did not solve, second LP did,
            # third search started but incomplete
        elif (
            status == xp.mip_lp_not_optimal
            or status == xp.mip_lp_optimal
            or status == xp.mip_no_sol_found
        ):
            results.solver.status = SolverStatus.aborted
            results.solver.termination_message = (
                "Model is loaded, but no solution information is available."
            )
            results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.unknown
        elif status == xp.mip_solution:  # some solution available
            results.solver.status = SolverStatus.warning
            results.solver.termination_message = (
                "Unable to satisfy optimality tolerances; a sub-optimal "
                "solution is available."
            )
            results.solver.termination_condition = TerminationCondition.other
            soln.status = SolutionStatus.feasible
        elif status == xp.mip_infeas:  # MIP proven infeasible
            results.solver.status = SolverStatus.warning
            results.solver.termination_message = "Model was proven to be infeasible"
            results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif status == xp.mip_optimal:  # optimal
            results.solver.status = SolverStatus.ok
            results.solver.termination_message = (
                "Model was solved to optimality (subject to tolerances), "
                "and an optimal solution is available."
            )
            results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif status == xp.mip_unbounded and mip_sols > 0:
            results.solver.status = SolverStatus.warning
            results.solver.termination_message = (
                "LP relaxation was proven to be unbounded, "
                "but a solution is available."
            )
            results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif status == xp.mip_unbounded and mip_sols <= 0:
            results.solver.status = SolverStatus.warning
            results.solver.termination_message = (
                "LP relaxation was proven to be unbounded."
            )
            results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        else:
            results.solver.status = SolverStatus.error
            results.solver.termination_message = (
                "Unhandled Xpress solve status (" + str(status) + ")"
            )
            results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error

        results.problem.upper_bound = None
        results.problem.lower_bound = None
        if xprob_attrs.objsense == 1.0:  # minimizing MIP
            try:
                results.problem.upper_bound = xprob_attrs.mipbestobjval
            except (XpressDirect.XpressException, AttributeError):
                pass
            try:
                results.problem.lower_bound = xprob_attrs.bestbound
            except (XpressDirect.XpressException, AttributeError):
                pass
        elif xprob_attrs.objsense == -1.0:  # maximizing MIP
            try:
                results.problem.upper_bound = xprob_attrs.bestbound
            except (XpressDirect.XpressException, AttributeError):
                pass
            try:
                results.problem.lower_bound = xprob_attrs.mipbestobjval
            except (XpressDirect.XpressException, AttributeError):
                pass

        return mip_sols > 0

    def _get_lp_results(self, results, soln):
        """Sets up `results` and `soln` and returns whether there is a solution
        to query.
        Returns `True` if a feasible solution is available, `False` otherwise.
        """
        xprob = self._solver_model
        xp = xpress
        xprob_attrs = xprob.attributes
        status = xprob_attrs.lpstatus
        if status == xp.lp_unstarted:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_message = (
                "Model is not loaded; no solution information is available."
            )
            results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.unknown
        elif status == xp.lp_optimal:
            results.solver.status = SolverStatus.ok
            results.solver.termination_message = (
                "Model was solved to optimality (subject to tolerances), "
                "and an optimal solution is available."
            )
            results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif status == xp.lp_infeas:
            results.solver.status = SolverStatus.warning
            results.solver.termination_message = "Model was proven to be infeasible"
            results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif status == xp.lp_cutoff:
            results.solver.status = SolverStatus.ok
            results.solver.termination_message = (
                "Optimal objective for model was proven to be worse than the "
                "cutoff value specified; a solution is available."
            )
            results.solver.termination_condition = TerminationCondition.minFunctionValue
            soln.status = SolutionStatus.optimal
        elif status == xp.lp_unfinished:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_message = (
                "Optimization was terminated by the user."
            )
            results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        elif status == xp.lp_unbounded:
            results.solver.status = SolverStatus.warning
            results.solver.termination_message = "Model was proven to be unbounded."
            results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif status == xp.lp_cutoff_in_dual:
            results.solver.status = SolverStatus.ok
            results.solver.termination_message = (
                "Xpress reported the LP was cutoff in the dual."
            )
            results.solver.termination_condition = TerminationCondition.minFunctionValue
            soln.status = SolutionStatus.optimal
        elif status == xp.lp_unsolved:
            results.solver.status = SolverStatus.error
            results.solver.termination_message = (
                "Optimization was terminated due to unrecoverable numerical "
                "difficulties."
            )
            results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        elif status == xp.lp_nonconvex:
            results.solver.status = SolverStatus.error
            results.solver.termination_message = (
                "Optimization was terminated because nonconvex quadratic data "
                "were found."
            )
            results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        else:
            results.solver.status = SolverStatus.error
            results.solver.termination_message = (
                "Unhandled Xpress solve status (" + str(status) + ")"
            )
            results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error

        results.problem.upper_bound = None
        results.problem.lower_bound = None
        try:
            results.problem.upper_bound = xprob_attrs.lpobjval
            results.problem.lower_bound = xprob_attrs.lpobjval
        except (XpressDirect.XpressException, AttributeError):
            pass

        # Not all solution information will be available in all cases, it is
        # up to the caller/user to check the actual status and figure which
        # of x, slack, duals, reduced costs are valid.
        return xprob_attrs.lpstatus in [
            xp.lp_optimal,
            xp.lp_cutoff,
            xp.lp_cutoff_in_dual,
        ]

    def _get_nlp_results(self, results, soln):
        """Sets up `results` and `soln` and returns whether there is a solution
        to query.
        Returns `True` if a feasible solution is available, `False` otherwise.
        """
        xprob = self._solver_model
        xp = xpress
        xprob_attrs = xprob.attributes
        solver = xprob_attrs.xslp_solverselected
        if solver == 2:
            # Under the hood we used the Xpress optimizer, i.e., the problem
            # was convex
            if (xprob_attrs.originalmipents > 0) or (xprob_attrs.originalsets > 0):
                return self._get_mip_results(results, soln)
            elif xprob_attrs.lpstatus and not xprob_attrs.xslp_nlpstatus:
                # If there is no NLP solver status, process the result
                # using the LP results processor.
                return self._get_lp_results(results, soln)

        # The problem was non-linear
        status = xprob_attrs.xslp_nlpstatus
        solstatus = xprob_attrs.xslp_solstatus
        have_soln = False
        optimal = False  # *globally* optimal?
        if status == xp.nlp_unstarted:
            results.solver.status = SolverStatus.unknown
            results.solver.termination_message = (
                "Non-convex model solve was not started"
            )
            results.solver.termination_condition = TerminationCondition.unknown
            soln.status = SolutionStatus.unknown
        elif status == xp.nlp_locally_optimal:
            # This is either xp.nlp_locally_optimal or xp.nlp_solution
            # we must look at the solstatus to figure out which
            if solstatus in [2, 3]:
                results.solver.status = SolverStatus.ok
                results.solver.termination_message = (
                    "Non-convex model was solved to local optimality"
                )
                results.solver.termination_condition = (
                    TerminationCondition.locallyOptimal
                )
                soln.status = SolutionStatus.locallyOptimal
            else:
                results.solver.status = SolverStatus.ok
                results.solver.termination_message = (
                    "Feasible solution found for non-convex model"
                )
                results.solver.termination_condition = TerminationCondition.feasible
                soln.status = SolutionStatus.feasible
            have_soln = True
        elif status == xp.nlp_globally_optimal:
            results.solver.status = SolverStatus.ok
            results.solver.termination_message = (
                "Non-convex model was solved to global optimality"
            )
            results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
            have_soln = True
            optimal = True
        elif status == xp.nlp_locally_infeasible:
            results.solver.status = SolverStatus.ok
            results.solver.termination_message = (
                "Non-convex model was proven to be locally infeasible"
            )
            results.solver.termination_condition = TerminationCondition.noSolution
            soln.status = SolutionStatus.unknown
        elif status == xp.nlp_infeasible:
            results.solver.status = SolverStatus.ok
            results.solver.termination_message = (
                "Non-convex model was proven to be infeasible"
            )
            results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif status == xp.nlp_unbounded:  # locally unbounded!
            results.solver.status = SolverStatus.ok
            results.solver.termination_message = "Non-convex model is locally unbounded"
            results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif status == xp.nlp_unfinished:
            results.solver.status = SolverStatus.ok
            results.solver.termination_message = (
                "Non-convex solve not finished (numerical issues?)"
            )
            results.solver.termination_condition = TerminationCondition.unknown
            soln.status = SolutionStatus.unknown
            have_soln = True
        else:
            results.solver.status = SolverStatus.error
            results.solver.termination_message = "Error for non-convex model: " + str(
                status
            )
            results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error

        results.problem.upper_bound = None
        results.problem.lower_bound = None
        try:
            if xprob_attrs.objsense > 0.0 or optimal:  # minimizing
                results.problem.upper_bound = xprob_attrs.xslp_objval
            if xprob_attrs.objsense < 0.0 or optimal:  # maximizing
                results.problem.lower_bound = xprob_attrs.xslp_objval
        except (XpressDirect.XpressException, AttributeError):
            pass

        return have_soln

    def _solve_model(self):
        if self._apply_warmstart:
            self._warm_start()

        xprob = self._solver_model

        is_mip = (xprob.attributes.mipents > 0) or (xprob.attributes.sets > 0)
        # Check for quadratic objective or quadratic constraints. If there are
        # any then we call nlpoptimize since that can handle non-convex
        # quadratics as well. In case of convex quadratics it will call
        # mipoptimize under the hood.
        if (xprob.attributes.qelems > 0) or (xprob.attributes.qcelems > 0):
            xprob.nlpoptimize("g" if is_mip else "")
            self._get_results = self._get_nlp_results
        elif is_mip:
            xprob.mipoptimize()
            self._get_results = self._get_mip_results
        else:
            xprob.lpoptimize()
            self._get_results = self._get_lp_results

        self._solver_model.postsolve()

    def _get_expr_from_pyomo_repn(self, repn, max_degree=2):
        referenced_vars = ComponentSet()

        degree = repn.polynomial_degree()
        if (degree is None) or (degree > max_degree):
            raise DegreeError(
                'XpressDirect does not support expressions of degree {0}.'.format(
                    degree
                )
            )

        # NOTE: xpress's python interface only allows for expressions
        #       with native numeric types. Others, like numpy.float64,
        #       will cause an exception when constructing expressions
        if len(repn.linear_vars) > 0:
            referenced_vars.update(repn.linear_vars)
            new_expr = xpress.Sum(
                float(coef) * self._pyomo_var_to_solver_var_map[var]
                for coef, var in zip(repn.linear_coefs, repn.linear_vars)
            )
        else:
            new_expr = 0.0

        for coef, (x, y) in zip(repn.quadratic_coefs, repn.quadratic_vars):
            new_expr += (
                float(coef)
                * self._pyomo_var_to_solver_var_map[x]
                * self._pyomo_var_to_solver_var_map[y]
            )
            referenced_vars.add(x)
            referenced_vars.add(y)

        new_expr += repn.constant

        return new_expr, referenced_vars

    def _get_expr_from_pyomo_expr(self, expr, max_degree=2):
        if max_degree == 2:
            repn = generate_standard_repn(expr, quadratic=True)
        else:
            repn = generate_standard_repn(expr, quadratic=False)

        try:
            xpress_expr, referenced_vars = self._get_expr_from_pyomo_repn(
                repn, max_degree
            )
        except DegreeError as e:
            msg = e.args[0]
            msg += '\nexpr: {0}'.format(expr)
            raise DegreeError(msg)

        return xpress_expr, referenced_vars

    def _xpress_lb_ub_from_var(self, var):
        if var.is_fixed():
            val = var.value
            return val, val
        if var.has_lb():
            lb = value(var.lb)
        else:
            lb = -xpress.infinity
        if var.has_ub():
            ub = value(var.ub)
        else:
            ub = xpress.infinity
        return lb, ub

    def _add_var(self, var):
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vartype = self._xpress_vartype_from_var(var)
        lb, ub = self._xpress_lb_ub_from_var(var)

        xpress_var = self._addVariable(
            self._solver_model, name=varname, lb=lb, ub=ub, vartype=vartype
        )

        ## bounds on binary variables don't seem to be set correctly
        ## by the method above
        if vartype == xpress.binary:
            if lb == ub:
                self._solver_model.chgbounds([xpress_var], ['B'], [lb])
            else:
                self._solver_model.chgbounds(
                    [xpress_var, xpress_var], ['L', 'U'], [lb, ub]
                )

        self._pyomo_var_to_solver_var_map[var] = xpress_var
        self._solver_var_to_pyomo_var_map[xpress_var] = var
        self._referenced_variables[var] = 0

    def _set_instance(self, model, kwds={}):
        self._range_constraints = set()
        DirectOrPersistentSolver._set_instance(self, model, kwds)
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = ComponentMap()
        try:
            if model.name is not None:
                self._solver_model = xpress.problem(name=model.name)
            else:
                self._solver_model = xpress.problem()
        except Exception:
            e = sys.exc_info()[1]
            msg = (
                "Unable to create Xpress model. "
                "Have you installed the Python "
                "bindings for Xpress?\n\n\t" + "Error message: {0}".format(e)
            )
            raise Exception(msg)
        self._add_block(model)

    def _add_block(self, block):
        DirectOrPersistentSolver._add_block(self, block)

    def _add_constraint(self, con):
        if not con.active:
            return None

        if self._skip_trivial_constraints and is_fixed(con.body):
            return None

        conname = self._symbol_map.getSymbol(con, self._labeler)

        if con._linear_canonical_form:
            xpress_expr, referenced_vars = self._get_expr_from_pyomo_repn(
                con.canonical_form(), self._max_constraint_degree
            )
        else:
            xpress_expr, referenced_vars = self._get_expr_from_pyomo_expr(
                con.body, self._max_constraint_degree
            )

        if con.has_lb():
            if not is_fixed(con.lower):
                raise ValueError(
                    "Lower bound of constraint {0} is not constant.".format(con)
                )
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError(
                    "Upper bound of constraint {0} is not constant.".format(con)
                )

        if con.equality:
            xpress_con = self._addConstraint(
                self._solver_model,
                body=xpress_expr,
                type=xpress.eq,
                rhs=value(con.lower),
                name=conname,
            )
        elif con.has_lb() and con.has_ub():
            xpress_con = self._addConstraint(
                self._solver_model,
                body=xpress_expr,
                type=xpress.rng,
                lb=value(con.lower),
                ub=value(con.upper),
                name=conname,
            )
            self._range_constraints.add(xpress_con)
        elif con.has_lb():
            xpress_con = self._addConstraint(
                self._solver_model,
                body=xpress_expr,
                type=xpress.geq,
                rhs=value(con.lower),
                name=conname,
            )
        elif con.has_ub():
            xpress_con = self._addConstraint(
                self._solver_model,
                body=xpress_expr,
                type=xpress.leq,
                rhs=value(con.upper),
                name=conname,
            )
        else:
            raise ValueError(
                "Constraint does not have a lower "
                "or an upper bound: {0} \n".format(con)
            )

        for var in referenced_vars:
            self._referenced_variables[var] += 1
        self._vars_referenced_by_con[con] = referenced_vars
        self._pyomo_con_to_solver_con_map[con] = xpress_con
        self._solver_con_to_pyomo_con_map[xpress_con] = con

    def _add_sos_constraint(self, con):
        if not con.active:
            return None

        conname = self._symbol_map.getSymbol(con, self._labeler)
        level = con.level
        if level not in [1, 2]:
            raise ValueError(
                "Solver does not support SOS level {0} constraints".format(level)
            )

        xpress_vars = []
        weights = []

        self._vars_referenced_by_con[con] = ComponentSet()

        if hasattr(con, 'get_items'):
            # aml sos constraint
            sos_items = list(con.get_items())
        else:
            # kernel sos constraint
            sos_items = list(con.items())

        for v, w in sos_items:
            self._vars_referenced_by_con[con].add(v)
            xpress_vars.append(self._pyomo_var_to_solver_var_map[v])
            self._referenced_variables[v] += 1
            weights.append(w)

        xpress_con = self._addSOS(
            self._solver_model, xpress_vars, weights, level, conname
        )
        self._pyomo_con_to_solver_con_map[con] = xpress_con
        self._solver_con_to_pyomo_con_map[xpress_con] = con

    def _xpress_vartype_from_var(self, var):
        """This function takes a pyomo variable and returns the appropriate
        xpress variable type

        :param var: pyomo.core.base.var.Var
        :return: xpress.continuous or xpress.binary or xpress.integer

        """
        if var.is_binary():
            vartype = xpress.binary
        elif var.is_integer():
            vartype = xpress.integer
        elif var.is_continuous():
            vartype = xpress.continuous
        else:
            raise ValueError(
                'Variable domain type is not recognized for {0}'.format(var.domain)
            )
        return vartype

    def _set_objective(self, obj):
        if self._objective is not None:
            for var in self._vars_referenced_by_obj:
                self._referenced_variables[var] -= 1
            self._vars_referenced_by_obj = ComponentSet()
            self._objective = None

        if obj.active is False:
            raise ValueError('Cannot add inactive objective to solver.')

        if obj.sense == minimize:
            sense = xpress.minimize
        elif obj.sense == maximize:
            sense = xpress.maximize
        else:
            raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

        xpress_expr, referenced_vars = self._get_expr_from_pyomo_expr(
            obj.expr, self._max_obj_degree
        )

        for var in referenced_vars:
            self._referenced_variables[var] += 1

        self._solver_model.setObjective(xpress_expr, sense=sense)
        self._objective = obj
        self._vars_referenced_by_obj = referenced_vars

    def _postsolve(self):
        # the only suffixes that we extract from XPRESS are
        # constraint duals, constraint slacks, and variable
        # reduced-costs. scan through the solver suffix list
        # and throw an exception if the user has specified
        # any others.
        extract_duals = False
        extract_slacks = False
        extract_reduced_costs = False
        for suffix in self._suffixes:
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
                raise RuntimeError(
                    "***The xpress_direct solver plugin cannot extract solution suffix="
                    + suffix
                )

        xprob = self._solver_model
        xp = xpress
        xprob_attrs = xprob.attributes

        ## XPRESS's status codes depend on this
        ## (number of integer vars > 0) or (number of special order sets > 0)
        is_mip = (xprob_attrs.mipents > 0) or (xprob_attrs.sets > 0)

        if is_mip:
            if extract_reduced_costs:
                logger.warning("Cannot get reduced costs for MIP.")
            if extract_duals:
                logger.warning("Cannot get duals for MIP.")
            extract_reduced_costs = False
            extract_duals = False

        self.results = SolverResults()
        soln = Solution()

        self.results.solver.name = XpressDirect._name
        self.results.solver.wallclock_time = self._opt_time

        if not hasattr(self, '_get_results'):
            raise RuntimeError(
                'Model was solved but `_get_results` property is not set'
            )
        have_soln = self._get_results(self.results, soln)
        self.results.problem.name = xprob_attrs.matrixname

        if xprob_attrs.objsense == 1.0:
            self.results.problem.sense = minimize
        elif xprob_attrs.objsense == -1.0:
            self.results.problem.sense = maximize
        else:
            raise RuntimeError(
                'Unrecognized Xpress objective sense: {0}'.format(xprob_attrs.objsense)
            )

        try:
            soln.gap = (
                self.results.problem.upper_bound - self.results.problem.lower_bound
            )
        except TypeError:
            soln.gap = None

        self.results.problem.number_of_constraints = (
            xprob_attrs.rows + xprob_attrs.sets + xprob_attrs.qconstraints
        )
        self.results.problem.number_of_nonzeros = xprob_attrs.elems
        self.results.problem.number_of_variables = xprob_attrs.cols
        self.results.problem.number_of_integer_variables = xprob_attrs.mipents
        self.results.problem.number_of_continuous_variables = (
            xprob_attrs.cols - xprob_attrs.mipents
        )
        self.results.problem.number_of_objectives = 1
        self.results.problem.number_of_solutions = xprob_attrs.mipsols if is_mip else 1

        # if a solve was stopped by a limit, we still need to check to
        # see if there is a solution available - this may not always
        # be the case, both in LP and MIP contexts.
        if self._save_results:
            # This code in this if statement is only needed for backwards
            # compatibility. It is more efficient to set _save_results to
            # False and use load_vars, load_duals, etc.
            if have_soln:
                soln_variables = soln.variable
                soln_constraints = soln.constraint

                if extract_duals or extract_slacks:
                    xpress_cons = list(self._solver_con_to_pyomo_con_map.keys())
                    for con in xpress_cons:
                        soln_constraints[con.name] = {}

                xpress_vars = list(self._solver_var_to_pyomo_var_map.keys())
                try:
                    var_vals = xprob.getSolution(xpress_vars)
                    if extract_slacks:
                        slacks = self._getSlacks(xprob, xpress_cons)
                except xpress.ModelError:
                    # Xpress 9.5.0 has new behavior for unbounded
                    # problems that have mipsols > 0.  Previously
                    # getSolution() would return a solution, but now
                    # raises a ModelError (even though the deprecated
                    # getmipsol() will return a solution).  We will try
                    # to fall back on the [deprecated] getmipsol(), but
                    # if it fails, we will raise the original exception.
                    try:
                        var_vals = []
                        slacks = [] if extract_slacks else None
                        xprob.getmipsol(var_vals, slacks)
                        fail = 0
                    except:
                        fail = 1
                    if fail:
                        raise

                for xpress_var, val in zip(xpress_vars, var_vals):
                    pyomo_var = self._solver_var_to_pyomo_var_map[xpress_var]
                    if self._referenced_variables[pyomo_var] > 0:
                        soln_variables[xpress_var.name] = {"Value": val}

                if extract_reduced_costs:
                    vals = self._getRedCosts(xprob, xpress_vars)
                    for xpress_var, val in zip(xpress_vars, vals):
                        pyomo_var = self._solver_var_to_pyomo_var_map[xpress_var]
                        if self._referenced_variables[pyomo_var] > 0:
                            soln_variables[xpress_var.name]["Rc"] = val

                if extract_duals:
                    vals = self._getDuals(xprob, xpress_cons)
                    for val, con in zip(vals, xpress_cons):
                        soln_constraints[con.name]["Dual"] = val

                if extract_slacks:
                    for con, val in zip(xpress_cons, slacks):
                        if con in self._range_constraints:
                            ## for xpress, the slack on a range constraint
                            ## is based on the upper bound
                            lb = con.lb
                            ub = con.ub
                            ub_s = val
                            expr_val = ub - ub_s
                            lb_s = lb - expr_val
                            if abs(ub_s) > abs(lb_s):
                                soln_constraints[con.name]["Slack"] = ub_s
                            else:
                                soln_constraints[con.name]["Slack"] = lb_s
                        else:
                            soln_constraints[con.name]["Slack"] = val

        elif self._load_solutions:
            if have_soln:
                self.load_vars()

                if extract_reduced_costs:
                    self._load_rc()

                if extract_duals:
                    self._load_duals()

                if extract_slacks:
                    self._load_slacks()

        self.results.solution.insert(soln)

        # finally, clean any temporary files registered with the temp file
        # manager, created populated *directly* by this plugin.
        TempfileManager.pop(remove=not self._keepfiles)
        return DirectOrPersistentSolver._postsolve(self)

    def warm_start_capable(self):
        return True

    def _warm_start(self):
        mipsolval = list()
        mipsolcol = list()
        for pyomo_var, xpress_var in self._pyomo_var_to_solver_var_map.items():
            if pyomo_var.value is not None:
                mipsolval.append(value(pyomo_var))
                mipsolcol.append(xpress_var)
        self._solver_model.addmipsol(mipsolval, mipsolcol)

    def _load_vars(self, vars_to_load=None):
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        xpress_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getSolution(xpress_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                var.set_value(val, skip_validation=True)

    def _load_rc(self, vars_to_load=None):
        if not hasattr(self._pyomo_model, 'rc'):
            self._pyomo_model.rc = Suffix(direction=Suffix.IMPORT)
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        rc = self._pyomo_model.rc
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        xpress_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._getRedCosts(self._solver_model, xpress_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                rc[var] = val

    def _load_duals(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'dual'):
            self._pyomo_model.dual = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        dual = self._pyomo_model.dual

        if cons_to_load is None:
            cons_to_load = con_map.keys()

        xpress_cons_to_load = [con_map[pyomo_con] for pyomo_con in cons_to_load]
        vals = self._getDuals(self._solver_model, xpress_cons_to_load)

        for pyomo_con, val in zip(cons_to_load, vals):
            dual[pyomo_con] = val

    def _load_slacks(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'slack'):
            self._pyomo_model.slack = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        slack = self._pyomo_model.slack

        if cons_to_load is None:
            cons_to_load = con_map.keys()

        xpress_cons_to_load = [con_map[pyomo_con] for pyomo_con in cons_to_load]
        vals = self._getSlacks(self._solver_model, xpress_cons_to_load)

        for pyomo_con, xpress_con, val in zip(cons_to_load, xpress_cons_to_load, vals):
            if xpress_con in self._range_constraints:
                ## for xpress, the slack on a range constraint
                ## is based on the upper bound
                lb = xpress_con.lb
                ub = xpress_con.ub
                ub_s = val
                expr_val = ub - ub_s
                lb_s = lb - expr_val
                if abs(ub_s) > abs(lb_s):
                    slack[pyomo_con] = ub_s
                else:
                    slack[pyomo_con] = lb_s
            else:
                slack[pyomo_con] = val

    def load_duals(self, cons_to_load=None):
        """Load the duals into the 'dual' suffix. The 'dual' suffix must live
        on the parent model.

        Parameters
        ----------
        cons_to_load: list of Constraint

        """
        self._load_duals(cons_to_load)

    def load_rc(self, vars_to_load=None):
        """Load the reduced costs into the 'rc' suffix. The 'rc' suffix must
        live on the parent model.

        Parameters
        ----------
        vars_to_load: list of Var

        """
        self._load_rc(vars_to_load)

    def load_slacks(self, cons_to_load=None):
        """Load the values of the slack variables into the 'slack' suffix. The
        'slack' suffix must live on the parent model.

        Parameters
        ----------
        cons_to_load: list of Constraint

        """
        self._load_slacks(cons_to_load)


# Note: because _finalize_xpress_import references XpressDirect, we need
# to make sure to not attempt the xpress import until after the
# XpressDirect class is fully declared.
_xpress_importer = _xpress_importer_class()
xpress, xpress_available = attempt_import(
    'xpress',
    error_message=_xpress_importer,
    # Other forms of exceptions can be thrown by the xpress python
    # import.  For example, an xpress.InterfaceError exception is thrown
    # if the Xpress license is not valid.  Unfortunately, you can't
    # import without a license, which means we can't test for that
    # explicit exception!
    catch_exceptions=(Exception,),
    importer=_xpress_importer,
    callback=_finalize_xpress_import,
)
