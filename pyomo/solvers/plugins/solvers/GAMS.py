#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from six import StringIO, iteritems, itervalues
from tempfile import mkdtemp

from pyomo.core.base import (Constraint, Suffix, Var, value,
                             Expression, Objective)
from pyomo.opt import ProblemFormat, SolverFactory
import os, sys, subprocess, math, logging, shutil

import pyomo.util.plugin
from pyomo.opt.base import IOptSolver
import pyutilib.services

from pyomo.opt.base.solvers import _extract_version
import pyutilib.subprocess
from pyutilib.misc import Options

from pyomo.core.kernel.component_block import IBlockStorage


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

    def __init__(self, **kwds):
        self._version = None

        self._capabilities = Options()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False

        self.options = Options()

        pyomo.util.plugin.Plugin.__init__(self, **kwds)

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

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        if not self.available(exception_flag=False):
            return _extract_version('')
        from gams import GamsWorkspace
        ws = GamsWorkspace()
        version = tuple(int(i) for i in ws._version.split('.'))
        while(len(version) < 4):
            version += (0,)
        version = version[:4]
        return version

    def version(self):
        """
        Returns a 4-tuple describing the solver executable version.
        """
        if self._version is None:
            self._version = self._get_version()
        return self._version

    def warm_start_capable(self):
        return True

    def solve(self, *args, **kwds):
        """
        Uses GAMS Python API. For installation help visit:
        https://www.gams.com/latest/docs/apis/examples_python/index.html

        tee=False:
            Output GAMS log to stdout.
        load_solutions=True:
            Does not support load_solutions=False.
        keepfiles=False:
            Keep temporary files. Equivalent of DebugLevel.KeepFiles.
            Summary of temp files can be found in _gams_py_gjo0.pf
        tmpdir=None:
            Specify directory path for storing temporary files.
            A directory will be created if one of this name doesn't exist.
        io_options:
            symbolic_solver_labels=False:
                Use full Pyomo component names rather than
                shortened symbols (slower, but useful for debugging).
                If passed as keyword to solve(), overrides io_options.
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

        assert len(args) == 1, 'Exactly one model must be passed '\
                               'to solve method of GAMSSolver.'
        model = args[0]

        warmstart      = kwds.pop("warmstart", True) # ignored
        load_solutions = kwds.pop("load_solutions", True)
        tee            = kwds.pop("tee", False)
        keepfiles      = kwds.pop("keepfiles", False)
        tmpdir         = kwds.pop("tmpdir", None)
        io_options     = kwds.pop("io_options", {})

        if "symbolic_solver_labels" in kwds:
            # If passed as keyword, override io_options
            symbolic_solver_labels = kwds.pop("symbolic_solver_labels")
            io_options.update(symbolic_solver_labels=symbolic_solver_labels)

        if len(kwds):
            raise ValueError(
                "GAMSSolver solve() passed unrecognized keyword args:\n\t" +
                "\n\t".join("%s = %s"
                            % (k,v) for k,v in iteritems(kwds)))

        if load_solutions is False:
            raise ValueError('GAMSSolver does not support '
                             'load_solutions=False.')

        # Create StringIO stream to pass to gams_writer, on which the
        # model file will be written. The writer also passes this StringIO
        # back, but output_file is defined in advance for clarity.
        output_file = StringIO()
        if isinstance(model, IBlockStorage):
            # Kernel blocks have slightly different write method
            symbolMap = model.write(filename=output_file,
                                    format=ProblemFormat.gams,
                                    **io_options)
        else:
            (_, smap_id) = model.write(filename=output_file,
                                       format=ProblemFormat.gams,
                                       io_options=io_options)
            symbolMap = model.solutions.symbol_map[smap_id]

        # IMPORTANT - only delete the whole tmpdir if the solver was the one
        # that made the directory. Otherwise, just delete the files the solver
        # made, if not keepfiles. That way the user can select a directory
        # they already have, like the current directory, without having to
        # worry about the rest of the contents of that directory being deleted.
        newdir = True
        if tmpdir is not None and os.path.exists(tmpdir):
            newdir = False

        ws = GamsWorkspace(debug=DebugLevel.KeepFiles if keepfiles
                           else DebugLevel.Off,
                           working_directory=tmpdir)
        
        t1 = ws.add_job_from_string(output_file.getvalue())

        try:
            t1.run(output=sys.stdout if tee else None)
        except GamsExceptionExecution:
            try:
                check_expr_evaluation(model, symbolMap, 'direct')
            finally:
                # Always name working directory or delete files,
                # regardless of any errors.
                if keepfiles:
                    print("\nGAMS WORKING DIRECTORY: %s\n" %
                          ws.working_directory)
                elif tmpdir is not None:
                    # Garbage collect all references to t1.out_db
                    # So that .gdx file can be deleted
                    t1 = rec = rec_lo = rec_hi = None
                    file_removal_gams_direct(tmpdir, newdir)
                raise
        except:
            if keepfiles:
                print("\nGAMS WORKING DIRECTORY: %s\n" % ws.working_directory)
            elif tmpdir is not None:
                # Garbage collect all references to t1.out_db
                # So that .gdx file can be deleted
                t1 = rec = rec_lo = rec_hi = None
                file_removal_gams_direct(tmpdir, newdir)
            raise

        has_dual = has_rc = False
        for suf in model.component_data_objects(Suffix, active=True):
            if isinstance(model, IBlockStorage):
                # Kernel suffix's import_enabled is property, not method
                if suf.name == 'dual' and suf.import_enabled:
                    has_dual = True
                elif suf.name == 'rc' and suf.import_enabled:
                    has_rc = True
            else:
                if suf.name == 'dual' and suf.import_enabled():
                    has_dual = True
                elif suf.name == 'rc' and suf.import_enabled():
                    has_rc = True

        for sym, ref in iteritems(symbolMap.bySymbol):
            obj = ref()
            if isinstance(model, IBlockStorage):
                # Kernel variables have no 'parent_component'
                if obj.ctype is not Var:
                    continue
            elif obj.parent_component().type() is not Var:
                continue
            rec = t1.out_db[sym].first_record()
            obj.value = rec.level
            if has_rc and not math.isnan(rec.marginal):
                # Do not set marginals to nan
                model.rc[obj] = rec.marginal

        if has_dual:
            for c in model.component_data_objects(Constraint, active=True):
                if c.body.is_fixed():
                    continue
                con = symbolMap.getSymbol(c)
                if c.equality:
                    rec = t1.out_db[con].first_record()
                    if not math.isnan(rec.marginal):
                        model.dual[c] = rec.marginal
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
                        model.dual[c] = marg
                    else:
                        # Solver didn't provide marginals,
                        # nothing else to do here
                        break

        if keepfiles:
            print("\nGAMS WORKING DIRECTORY: %s\n" % ws.working_directory)
        elif tmpdir is not None:
            # Garbage collect all references to t1.out_db
            # So that .gdx file can be deleted
            t1 = rec = rec_lo = rec_hi = None
            file_removal_gams_direct(tmpdir, newdir)

        return None


class GAMSShell(pyomo.util.plugin.Plugin):
    """A generic interface to GAMS solvers"""
    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('_gams_shell', doc='The GAMS modeling language')

    def __init__(self, **kwds):
        self._version = None

        self._capabilities = Options()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False

        self.options = Options()

        pyomo.util.plugin.Plugin.__init__(self, **kwds)

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
 
    def _default_executable(self):
        executable = pyutilib.services.registered_executable("gams")
        if executable is None:
            logger.warning("Could not locate the 'gams' executable, "
                           "which is required for solver gams")
            self.enable = False
            return None
        return executable.get_path()

    def executable(self):
        """
        Returns the executable used by this solver.
        """
        return self._default_executable()

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        solver_exec = self.executable()

        if solver_exec is None:
            return _extract_version('')
        else:
            results = pyutilib.subprocess.run([solver_exec])
            return _extract_version(results[1])

    def version(self):
        """
        Returns a 4-tuple describing the solver executable version.
        """
        if self._version is None:
            self._version = self._get_version()
        return self._version

    def warm_start_capable(self):
        return True

    def solve(self, *args, **kwds):
        """
        Uses command line to call GAMS.
        Uses temp file GAMS_results.dat for parsing result data.

        tee=False:
            Output GAMS log to stdout.
        load_solutions=True:
            Does not support load_solutions=False.
        keepfiles=False:
            Keep temporary files.
        tmpdir=None:
            Specify directory path for storing temporary files.
            A directory will be created if one of this name doesn't exist.
        io_options:
            symbolic_solver_labels=False:
                Use full Pyomo component names rather than
                shortened symbols (slower, but useful for debugging).
                If passed as keyword to solve(), overrides io_options.
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

        assert len(args) == 1, 'Exactly one model must be passed '\
                               'to solve method of GAMSSolver.'
        model = args[0]

        warmstart      = kwds.pop("warmstart", True) # ignored
        load_solutions = kwds.pop("load_solutions", True)
        tee            = kwds.pop("tee", False)
        keepfiles      = kwds.pop("keepfiles", False)
        tmpdir         = kwds.pop("tmpdir", None)
        io_options     = kwds.pop("io_options", {})

        if "symbolic_solver_labels" in kwds:
            # If passed as keyword, override io_options
            symbolic_solver_labels = kwds.pop("symbolic_solver_labels")
            io_options.update(symbolic_solver_labels=symbolic_solver_labels)

        if len(kwds):
            raise ValueError(
                "GAMSSolver solve() passed unrecognized keyword args:\n\t" +
                "\n\t".join("%s = %s"
                            % (k,v) for k,v in iteritems(kwds)))

        if load_solutions is False:
            raise ValueError('GAMSSolver does not support '
                             'load_solutions=False.')

        # IMPORTANT - only delete the whole tmpdir if the solver was the one
        # that made the directory. Otherwise, just delete the files the solver
        # made, if not keepfiles. That way the user can select a directory
        # they already have, like the current directory, without having to
        # worry about the rest of the contents of that directory being deleted.
        newdir = False
        if tmpdir is None:
            tmpdir = mkdtemp()
            newdir = True
        elif not os.path.exists(tmpdir):
            # makedirs creates all necessary intermediate directories in order
            # to create the path to tmpdir, if they don't already exist.
            # However, if keepfiles is False, we only delete the final folder,
            # leaving the rest of the intermediate ones.
            os.makedirs(tmpdir)
            newdir = True

        output_filename = os.path.join(tmpdir, 'model.gms')
        lst_filename = os.path.join(tmpdir, 'output.lst')
        results_filename = os.path.join(tmpdir, 'results.dat')

        io_options['put_results'] = results_filename

        if isinstance(model, IBlockStorage):
            # Kernel blocks have slightly different write method
            symbolMap = model.write(filename=output_filename,
                                    format=ProblemFormat.gams,
                                    **io_options)
        else:
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

            if keepfiles:
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
            if not keepfiles:
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
            if isinstance(model, IBlockStorage):
                # Kernel suffix's import_enabled is property, not method
                if suf.name == 'dual' and suf.import_enabled:
                    has_dual = True
                elif suf.name == 'rc' and suf.import_enabled:
                    has_rc = True
            else:
                if suf.name == 'dual' and suf.import_enabled():
                    has_dual = True
                elif suf.name == 'rc' and suf.import_enabled():
                    has_rc = True

        for sym, ref in iteritems(symbolMap.bySymbol):
            obj = ref()
            if isinstance(model, IBlockStorage):
                # Kernel variables have no 'parent_component'
                if obj.ctype is not Var:
                    continue
            elif obj.parent_component().type() is not Var:
                continue
            rec = soln[sym]
            obj.value = float(rec[0])
            if has_rc:
                try:
                    model.rc[obj] = float(rec[1])
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
                        model.dual[c] = float(rec[1])
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
                        model.dual[c] = marg
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
        for ref in itervalues(symbolMap.bySymbol):
            obj = ref()
            if isinstance(model, IBlockStorage):
                if obj.ctype is Expression:
                    check_expr(obj.expr, obj.name, solver_io)
            elif obj.parent_component().type() is Expression:
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

def file_removal_gams_direct(tmpdir, newdir):
    if newdir:
        shutil.rmtree(tmpdir)
    else:
        os.remove(os.path.join(tmpdir, '_gams_py_gjo0.gms'))
        os.remove(os.path.join(tmpdir, '_gams_py_gjo0.lst'))
        os.remove(os.path.join(tmpdir, '_gams_py_gdb0.gdx'))
        # .pf file is not made when DebugLevel is Off
