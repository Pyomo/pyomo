#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base import Constraint, Suffix, Var, value
from pyomo.opt import ProblemFormat, SolverFactory
import os, sys, subprocess, pipes, math, logging

import pyomo.util.plugin
from pyomo.opt.base import IOptSolver
import pyutilib.services


logger = logging.getLogger('pyomo.solvers')

pyutilib.services.register_executable(name="gams")

class GAMSSolver(pyomo.util.plugin.Plugin):
    """A generic interface to GAMS solvers"""
    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('gams', doc='The GAMS modeling language')

    def __new__(cls, *args, **kwds):
        try:
            mode = kwds['solver_io']
            if mode is None:
                mode = 'direct'
            del kwds['solver_io']
        except KeyError:
            mode = 'direct'

        if mode == 'direct':
            return SolverFactory('_gams_direct', **kwds)
        if mode == 'shell':
            return SolverFactory('_gams_shell', **kwds)
        else:
            logger.error('Unknown IO type: %s' % mode)
            return


class GAMSDirect(pyomo.util.plugin.Plugin):
    """A generic interface to GAMS solvers"""
    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('_gams_direct', doc='The GAMS modeling language')

    def available(self, exception_flag=True):
        try:
            from gams import GamsWorkspace, DebugLevel
            return True
        except ImportError:
            return False
 
    def solve(self,
              model,
              tee=False,
              load_model=True,
              keep_files=False,
              io_options={}):
        """
        Uses GAMS Python API. For installation help visit:
        https://www.gams.com/latest/docs/apis/examples_python/index.html

        tee=False:
            Output GAMS log to stdout.
        load_model=True:
            Load results back into pyomo model.
        keep_files=False:
            Keep temporary files in folder '_GAMSSolver_files'.
            Equivalent of DebugLevel.KeepFiles.
            Summary of temp files can be found in _gams_py_gjo0.pf
        io_options:
            symbolic_solver_labels=False:
                Use full Pyomo component names rather than
                shortened symbols (slower, but useful for debugging).
            labeler=None:
                Custom labeler option. Incompatible with symbolic_solver_labels.
            solver=None:
                If None, GAMS will use default solver for model type.
            mtype=None:
                Model type. If None, will chose from lp, nlp, mip, and minlp.
            add_options:
                List of additional lines to write directly
                into model file before the solve statement.
                For model attributes, <model name> = GAMS_MODEL
        """

        from gams import GamsWorkspace, DebugLevel
        from gams.workspace import GamsExceptionExecution

        var_list = []
        io_options['var_list'] = var_list
        io_options['stringio'] = True

        (output_file, smap_id) = model.write(format=ProblemFormat.gams,
                                                 io_options=io_options)
        symbolMap = model.solutions.symbol_map[smap_id]

        # If keeping files, set dir to current dir
        # All tmp files will be created and kept there
        # Otherwise dir is in /tmp/ folder and deleted after
        if keep_files:
            folder = os.path.join(os.getcwd(),'_GAMSSolver_files')
            ws = GamsWorkspace(working_directory=folder,
                               debug=DebugLevel.KeepFiles)
        else:
            ws = GamsWorkspace()
        
        t1 = ws.add_job_from_string(output_file.getvalue())

        try:
            if tee:
                t1.run(output=sys.stdout)
            else:
                t1.run()
        except GamsExceptionExecution:
            try:
                check_expr_evaluation(model, var_list, symbolMap, 'direct')
            finally:
                raise

        has_dual = has_rc = False
        for suf in model.component_data_objects(Suffix, active=True):
            if (suf.name == 'dual' and suf.import_enabled()):
                has_dual = True
            elif (suf.name == 'rc' and suf.import_enabled()):
                has_rc = True

        if load_model:
            for var in var_list:
                v = symbolMap.getObject(var)
                if not v.is_expression():
                    rec = t1.out_db[var].first_record()
                    if v.is_binary() or v.is_integer():
                        v.set_value(int(rec.level))
                    else:
                        v.set_value(rec.level)
                    if has_rc and not math.isnan(rec.marginal):
                        # Do not set marginals to nan
                        model.rc.set_value(v, rec.marginal)

        if load_model and has_dual:
            for c in model.component_data_objects(Constraint, active=True):
                if c.body.is_fixed():
                    continue
                con = symbolMap.getSymbol(c)
                if c.equality:
                    rec = t1.out_db[con].first_record()
                    if not math.isnan(rec.marginal):
                        model.dual.set_value(c, rec.marginal)
                    else:
                        # Solver didn't provide marginals,
                        # nothing else to do here
                        break
                else:
                    # Inequality, assume if 2-sided that only
                    # one side's marginal is nonzero
                    marg = 0
                    if c.lower is not None:
                        rec_lo = t1.out_db[con + '_lo'].first_record()
                        marg += rec_lo.marginal
                    if c.upper is not None:
                        rec_hi = t1.out_db[con + '_hi'].first_record()
                        marg += rec_hi.marginal
                    if not math.isnan(rec.marginal):
                        model.dual.set_value(c, marg)
                    else:
                        # Solver didn't provide marginals,
                        # nothing else to do here
                        break
        return None


class GAMSShell(pyomo.util.plugin.Plugin):
    """Shell interface to GAMS solvers"""
    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('_gams_shell', doc='The GAMS modeling language')

    def available(self, exception_flag=True):
        return pyutilib.services.registered_executable("gams") is not None
 
    def solve(self,
              model,
              tee=False,
              load_model=True,
              keep_files=False,
              io_options={}):
        """
        Uses command line to call GAMS.
        Uses temp file GAMS_results.dat for parsing result data.

        tee=False:
            Output GAMS log to stdout.
        load_model=True:
            Load results back into pyomo model.
        keep_files=False:
            Keep .gms and .lst files in current directory.
        io_options:
            symbolic_solver_labels=False:
                Use full Pyomo component names rather than
                shortened symbols (slower, but useful for debugging).
            labeler=None:
                Custom labeler option. Incompatible with symbolic_solver_labels.
            solver=None:
                If None, GAMS will use default solver for model type.
            mtype=None:
                Model type. If None, will chose from lp, nlp, mip, and minlp.
            add_options:
                List of additional lines to write directly
                into model file before the solve statement.
                For model attributes, <model name> = GAMS_MODEL
        """

        var_list = []
        io_options['var_list'] = var_list
        io_options['put_results'] = True

        (output_filename, smap_id) = model.write(format=ProblemFormat.gams,
                                                 io_options=io_options)
        symbolMap = model.solutions.symbol_map[smap_id]

        command = "gams " + pipes.quote(output_filename)
        if tee:
            rc = subprocess.call(command, shell=True)
        else:
            rc = subprocess.call(command + " lo=0", shell=True)

        if rc == 1 or rc == 127:
            raise RuntimeError("Command 'gams' was not recognized")
        elif rc == 3:
            # Run check_expr_evaluation, which errors if necessary
            check_expr_evaluation(model, var_list, symbolMap, 'shell')
            # If nothing was raised, raise the regular RuntimeError
            raise RuntimeError("GAMS encountered an error during solve. "
                               "Check listing file for details.")
        elif rc != 0:
            raise RuntimeError("GAMS encountered an error during solve. "
                               "Check listing file for details.")

        with open('GAMS_results.dat', 'r') as results_file:
            results_text = results_file.read()
        soln = dict()
        for line in results_text.splitlines():
            items = line.split()
            soln[items[0]] = (items[1], items[2])
        os.remove('GAMS_results.dat')

        if not keep_files:
            os.remove(output_filename)
            os.remove(os.path.splitext(output_filename)[0] + '.lst')

        has_dual = has_rc = False
        for suf in model.component_data_objects(Suffix, active=True):
            if (suf.name == 'dual' and suf.import_enabled()):
                has_dual = True
            elif (suf.name == 'rc' and suf.import_enabled()):
                has_rc = True

        if load_model:
            for var in var_list:
                rec = soln[var]
                v = symbolMap.getObject(var)
                if not v.is_expression():
                    if v.is_binary() or v.is_integer():
                        v.set_value(int(float(rec[0])))
                    else:
                        v.set_value(float(rec[0]))
                    if has_rc:
                        try:
                            model.rc.set_value(v, float(rec[1]))
                        except ValueError:
                            # Solver didn't provide marginals
                            pass

        if load_model and has_dual:
            for c in model.component_data_objects(Constraint, active=True):
                if c.body.is_fixed():
                    continue
                con = symbolMap.getSymbol(c)
                if c.equality:
                    rec = soln[con]
                    try:
                        model.dual.set_value(c, float(rec[1]))
                    except ValueError:
                        # Solver didn't provide marginals
                        # nothing else to do here
                        break
                else:
                    # Inequality, assume if 2-sided that only
                    # one side's marginal is nonzero
                    marg = 0
                    if c.lower is not None:
                        rec_lo = soln[con + '_lo']
                        try:
                            marg += float(rec_lo[1])
                        except ValueError:
                            # Solver didn't provide marginals
                            marg = float('nan')
                    if c.upper is not None:
                        rec_hi = soln[con + '_hi']
                        try:
                            marg += float(rec_hi[1])
                        except ValueError:
                            # Solver didn't provide marginals
                            marg = float('nan')
                    if not math.isnan(marg):
                        model.dual.set_value(c, marg)
                    else:
                        # Solver didn't provide marginals
                        # nothing else to do here
                        break
        return None


def check_expr_evaluation(model, var_list, symbolMap, solver_io):
    try:
        # Temporarily initialize uninitialized variables in order to call
        # value() on each expression to check domain violations
        uninit_vars = list()
        for var in model.component_data_objects(Var, active=True):
            if var.value is None:
                uninit_vars.append(var)
                var.value = 0

        # Constraints
        for con in model.component_data_objects(Constraint, active=True):
            if con.body.is_fixed():
                continue
            check_expr(con.body, con.name, solver_io)

        # Objective
        obj = list(model.component_data_objects(Objective, active=True))
        assert len(obj) == 1, "GAMS writer can only take 1 active objective"
        obj = obj[0]
        check_expr(obj.expr, obj.name, solver_io)

        # Expressions
        # Iterate through var_list in case for some reason model has Expressions
        # that do not appear in any constraints or the objective, since GAMS
        # never sees those anyway so they should be skipped
        for var in var_list:
            v = symbolMap.getObject(var)
            if v.is_expression:
                check_expr(v.expr, v.name, solver_io)
    finally:
        # Return uninitialized variables to None
        for var in uninit_vars:
            var.value = None

def check_expr(expr, name, solver_io):
    # Check if GAMS will encounter domain violations in presolver
    # operations at current values, which are None (0) by default
    # Used to handle log and log10 violations, for example
    try:
        value(expr)
    except ValueError:
        logger.warning("While evaluating value(%s), GAMS solver encountered "
                       "an error.\nGAMS requires that all equations and "
                       "expressions evaluate at initial values.\n"
                       "Ensure variable values do not violate any domains "
                       "(are you using log or log10?)" % name)
        if solver_io == 'shell':
            # For shell, there is no previous exception to worry about
            # overwriting, so raise the ValueError.
            # But for direct, the GamsExceptionExecution will be raised.
            raise
