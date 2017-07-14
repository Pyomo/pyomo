#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from six import StringIO
from tempfile import mkdtemp

from pyomo.core.base import Constraint, Suffix, Var, value, Expression
from pyomo.opt import ProblemFormat, SolverFactory
import os, sys, subprocess, pipes, math, logging, shutil

import pyomo.util.plugin
from pyomo.opt.base import IOptSolver
import pyutilib.services


logger = logging.getLogger('pyomo.solvers')

pyutilib.services.register_executable(name="gams")

class GAMSSolver(pyomo.util.plugin.Plugin):
    """
    A generic interface to GAMS solvers

    Pass solver_io keyword arg to SolverFactory to choose solver mode:
        solver_io='direct' or 'python' to use GAMS Python API
            Requires installation, visit this url for help:
            https://www.gams.com/latest/docs/apis/examples_python/index.html
        solver_io='shell' or 'gms' to use command line to call gams
            Requires the gams executable be on your system PATH
    """
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

        if mode == 'direct' or mode == 'python':
            return SolverFactory('_gams_direct', **kwds)
        if mode == 'shell' or mode == 'gms':
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

    def available(self, exception_flag=True):
        """True if the solver is available"""
        try:
            from gams import GamsWorkspace, DebugLevel
            return True
        except ImportError as e:
            if exception_flag is False:
                return False
            else:
                raise ImportError("Import of gams failed - GAMS direct "
                                  "solver functionality is not available.\n"
                                  "GAMS message: %s" % e)

    def solve(self,
              model,
              tee=False,
              load_model=True,
              keep_files=False,
              tmpdir=None,
              io_options={}):
        """
        Uses GAMS Python API. For installation help visit:
        https://www.gams.com/latest/docs/apis/examples_python/index.html

        tee=False:
            Output GAMS log to stdout.
        load_model=True:
            Does not support load_model=False.
        keep_files=False:
            Keep temporary files in folder '_GAMSSolver_files'.
            Equivalent of DebugLevel.KeepFiles.
            Summary of temp files can be found in _gams_py_gjo0.pf
        tmpdir=None:
            Specify directory path for storing temporary files.
            A directory will be created if one of this name doesn't exist.
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

        # Make sure available() doesn't crash
        self.available()

        from gams import GamsWorkspace, DebugLevel
        from gams.workspace import GamsExceptionExecution

        assert load_model == True, 'GAMSSolver does not support '\
                                   'load_model=False.'

        # Create StringIO stream to pass to gams_writer, on which the
        # model file will be written. The writer also passes this StringIO
        # back, but output_file is defined in advance for clarity.
        output_file = StringIO()
        (_, smap_id) = model.write(filename=output_file,
                                   format=ProblemFormat.gams,
                                   io_options=io_options)
        symbolMap = model.solutions.symbol_map[smap_id]

        # IMPORTANT - only delete the whole tmpdir if the solver was the one
        # that made the directory. Otherwise, just delete the files the solver
        # made, if not keep_files. That way the user can select a directory
        # they already have, like the current directory, without having to
        # worry about the rest of the contents of that directory being deleted.
        newdir = True
        if tmpdir is not None and os.path.exists(tmpdir):
            newdir = False

        ws = GamsWorkspace(debug=DebugLevel.KeepFiles if keep_files
                           else DebugLevel.Off,
                           working_directory=tmpdir)
        
        t1 = ws.add_job_from_string(output_file.getvalue())

        try:
            t1.run(output=sys.stdout if tee else None)
        except GamsExceptionExecution:
            try:
                check_expr_evaluation(model, symbolMap, 'direct')
            finally:
                raise
        finally:
            if keep_files:
                print("\nGAMS WORKING DIRECTORY: %s\n" % ws.working_directory)
            elif tmpdir is not None:
                if newdir:
                    shutil.rmtree(tmpdir)
                else:
                    os.remove(os.path.join(tmpdir, '_gams_py_gjo0.gms'))
                    os.remove(os.path.join(tmpdir, '_gams_py_gjo0.lst'))
                    os.remove(os.path.join(tmpdir, '_gams_py_gdb0.gdx'))
                    # .pf file is not made when DebugLevel is Off

        has_dual = has_rc = False
        for suf in model.component_data_objects(Suffix, active=True):
            if (suf.name == 'dual' and suf.import_enabled()):
                has_dual = True
            elif (suf.name == 'rc' and suf.import_enabled()):
                has_rc = True

        for sym, ref in symbolMap.bySymbol.iteritems():
            obj = ref()
            if not obj.parent_component().type() is Var:
                continue
            rec = t1.out_db[sym].first_record()
            obj.value = rec.level
            if has_rc and not math.isnan(rec.marginal):
                # Do not set marginals to nan
                model.rc.set_value(obj, rec.marginal)

        if has_dual:
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
    """A generic interface to GAMS solvers"""
    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('_gams_shell', doc='The GAMS modeling language')

    def available(self, exception_flag=True):
        """True if the solver is available"""
        exe = pyutilib.services.registered_executable("gams")
        if exception_flag is False:
            return exe is not None
        else:
            if exe is not None:
                return True
            else:
                raise NameError(
                    "No 'gams' command found on system PATH - GAMS shell "
                    "solver functionality is not available.")
 
    def solve(self,
              model,
              tee=False,
              load_model=True,
              keep_files=False,
              tmpdir=None,
              io_options={}):
        """
        Uses command line to call GAMS.
        Uses temp file GAMS_results.dat for parsing result data.

        tee=False:
            Output GAMS log to stdout.
        load_model=True:
            Does not support load_model=False.
        keep_files=False:
            Keep .gms and .lst files in current directory.
        tmpdir=None:
            Specify directory path for storing temporary files.
            A directory will be created if one of this name doesn't exist.
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

        # Make sure available() doesn't crash
        self.available()

        assert load_model == True, 'GAMSSolver does not support '\
                                   'load_model=False.'

        # IMPORTANT - only delete the whole tmpdir if the solver was the one
        # that made the directory. Otherwise, just delete the files the solver
        # made, if not keep_files. That way the user can select a directory
        # they already have, like the current directory, without having to
        # worry about the rest of the contents of that directory being deleted.
        newdir = False
        if tmpdir is None:
            tmpdir = mkdtemp()
            newdir = True
        elif not os.path.exists(tmpdir):
            # makedirs creates all necessary intermediate directories in order
            # to create the path to tmpdir, if they don't already exist.
            # However, if keep_files is False, we only delete the final folder,
            # leaving the rest of the intermediate ones.
            os.makedirs(tmpdir)
            newdir = True

        output_filename = os.path.join(tmpdir, 'model.gms')
        lst_filename = os.path.join(tmpdir, 'output.lst')
        results_filename = os.path.join(tmpdir, 'results.dat')

        io_options['put_results'] = results_filename

        (_, smap_id) = model.write(filename=output_filename,
                                   format=ProblemFormat.gams,
                                   io_options=io_options)
        symbolMap = model.solutions.symbol_map[smap_id]

        exe = pyutilib.services.registered_executable("gams")
        command = [exe.get_path(), output_filename, 'o=' + lst_filename]
        if not tee:
            command.append("lo=0")

        try:
            rc = subprocess.call(command)

            if keep_files:
                print("\nGAMS WORKING DIRECTORY: %s\n" % tmpdir)

            if rc == 1 or rc == 127:
                raise RuntimeError("Command 'gams' was not recognized")
            elif rc != 0:
                if rc == 3:
                    # Execution Error
                    # Run check_expr_evaluation, which errors if necessary
                    check_expr_evaluation(model, symbolMap, 'shell')
                # If nothing was raised, or for all other cases, raise this
                raise RuntimeError("GAMS encountered an error during solve. "
                                   "Check listing file for details.")

            with open(results_filename, 'r') as results_file:
                results_text = results_file.read()
        finally:
            if not keep_files:
                if newdir:
                    shutil.rmtree(tmpdir)
                else:
                    os.remove(output_filename)
                    os.remove(lst_filename)
                    os.remove(results_filename)

        soln = dict()
        # Skip first line of explanatory text
        for line in results_text.splitlines()[1:]:
            items = line.split()
            soln[items[0]] = (items[1], items[2])

        has_dual = has_rc = False
        for suf in model.component_data_objects(Suffix, active=True):
            if (suf.name == 'dual' and suf.import_enabled()):
                has_dual = True
            elif (suf.name == 'rc' and suf.import_enabled()):
                has_rc = True

        for sym, ref in symbolMap.bySymbol.iteritems():
            obj = ref()
            if not obj.parent_component().type() is Var:
                continue
            rec = soln[sym]
            obj.value = float(rec[0])
            if has_rc:
                try:
                    model.rc.set_value(obj, float(rec[1]))
                except ValueError:
                    # Solver didn't provide marginals
                    pass

        if has_dual:
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


def check_expr_evaluation(model, symbolMap, solver_io):
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
        # Iterate through symbolMap in case for some reason model has
        # Expressions that do not appear in any constraints or the objective,
        # since GAMS never sees those anyway so they should be skipped
        for ref in symbolMap.bySymbol.itervalues():
            obj = ref()
            if obj.type() is Expression:
                check_expr(obj.expr, obj.name, solver_io)
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
