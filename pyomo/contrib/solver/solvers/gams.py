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
import shutil
import subprocess
import datetime
from io import StringIO
from typing import Mapping, Optional, Sequence, Tuple
import sys

from pyomo.common.fileutils import Executable, ExecutableData
from pyomo.common.dependencies import pathlib
from pyomo.common.config import (
    ConfigValue,
    ConfigDict,
    document_configdict,
    Path,
    document_class_CONFIG,
)
from pyomo.common.modeling import NOTSET
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base import Constraint, Var, value, Objective
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.solver.common.base import SolverBase, Availability
from pyomo.contrib.solver.common.config import SolverConfig
from pyomo.opt.results import SolverStatus, TerminationCondition
from pyomo.contrib.solver.common.results import (
    legacy_termination_condition_map,
    Results,
    SolutionStatus,
)
from pyomo.contrib.solver.solvers.gms_sol_reader import GMSSolutionLoader

import pyomo.core.base.suffix
from pyomo.common.tee import TeeStream
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.expr.numvalue import value
from pyomo.core.base.suffix import Suffix
from pyomo.common.errors import (
    ApplicationError,
    DeveloperError,
    InfeasibleConstraintException,
)

from pyomo.repn.plugins.gams_writer_v2 import GAMSWriterInfo, GAMSWriter

logger = logging.getLogger(__name__)

from pyomo.common.dependencies import attempt_import
import struct


def _gams_importer():
    try:
        import gams.core.gdx as gdx

        return gdx
    except ImportError:
        try:
            # fall back to the pre-GAMS-45.0 API
            import gdxcc

            return gdxcc
        except:
            # suppress the error from the old API and reraise the current API import error
            pass
        raise


gdxcc, gdxcc_available = attempt_import('gdxcc', importer=_gams_importer)


@document_configdict()
class GAMSConfig(SolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        self.executable: ExecutableData = self.declare(
            'executable',
            ConfigValue(
                domain=Executable,
                default='gams',
                description="Executable for gams. Defaults to searching the "
                "``PATH`` for the first available ``gams``.",
            ),
        )
        self.logfile: str = self.declare(
            'logfile',
            ConfigValue(
                domain=Path(),
                default=None,
                description="Filename to output GAMS log to a file.",
            ),
        )
        self.writer_config: ConfigDict = self.declare(
            'writer_config', GAMSWriter.CONFIG()
        )

        # NOTE: Taken from the lp_writer
        self.declare(
            'row_order',
            ConfigValue(
                default=None,
                description='Preferred constraint ordering',
                doc="""
                To use with ordered_active_constraints function.""",
            ),
        )


class GAMSResults(Results):
    def __init__(self):
        super().__init__()
        self.return_code: ConfigDict = self.declare(
            'return_code',
            ConfigValue(default=None, description="Return code from the GAMS solver."),
        )
        self.gams_termination_condition: ConfigDict = self.declare(
            'gams_termination_condition',
            ConfigValue(
                default=None,
                description="Include additional TerminationCondition domain.",
            ),
        )
        self.gams_solver_status: ConfigDict = self.declare(
            'gams_solver_status',
            ConfigValue(
                default=None, description="Include additional SolverStatus domain."
            ),
        )


@document_class_CONFIG(methods=['solve'])
class GAMS(SolverBase):
    CONFIG = GAMSConfig()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._writer = GAMSWriter()
        self._available_cache = NOTSET
        self._version_cache = NOTSET

    def available(
        self, config: Optional[GAMSConfig] = None, rehash: bool = False
    ) -> Availability:
        if config is None:
            config = self.config

        pth = config.executable.path()

        if pth is None:
            self._available_cache = (None, Availability.NotFound)
        else:
            self._available_cache = (pth, Availability.FullLicense)
        if self._available_cache is not NOTSET and rehash == False:
            return self._available_cache[1]
        else:
            raise NotImplementedError('feature for rehash is WIP')
            # Executable(pth).available()
            # Executable(pth).rehash()

    def _run_simple_model(self, config, n):
        solver_exec = config.executable.path()
        if solver_exec is None:
            return False
        with TempfileManager.new_context() as tempfile:
            tmpdir = tempfile.mkdtemp()
            test = os.path.join(tmpdir, 'test.gms')
            with open(test, 'w') as FILE:
                FILE.write(self._simple_model(n))
            result = subprocess.run(
                [solver_exec, test, "curdir=" + tmpdir, 'lo=0'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return not result.returncode

    def _simple_model(self, n):
        return """
            option limrow = 0;
            option limcol = 0;
            option solprint = off;
            set I / 1 * %s /;
            variables ans;
            positive variables x(I);
            equations obj;
            obj.. ans =g= sum(I, x(I));
            model test / all /;
            solve test using lp minimizing ans;
            """ % (
            n,
        )

    def version(
        self, config: Optional[GAMSConfig] = None, rehash: bool = False
    ) -> Optional[Tuple[int, int, int]]:

        if config is None:
            config = self.config
        pth = config.executable.path()

        if pth is None:
            self._version_cache = (None, None)
        else:
            cmd = [pth, "audit", "lo=3"]
            subprocess_results = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                check=False,
            )
            version = subprocess_results.stdout.splitlines()[0]
            version = [char for char in version.split(' ') if len(char) > 0][1]
            version = tuple(int(i) for i in version.split('.'))
            self._version_cache = (pth, version)

        if self._version_cache is not NOTSET and rehash == False:
            return self._version_cache[1]

        else:
            raise NotImplementedError('feature for rehash is WIP')

    def _rewrite_path_win8p3(self, path):
        """
        Return the 8.3 short path on Windows; unchanged elsewhere.

        This change is in response to Pyomo/pyomo#3579 which reported
        that GAMS (direct) fails on Windows if there is a space in
        the path. This utility converts paths to their 8.3 short-path version
        (which never have spaces).
        """
        if not sys.platform.startswith("win"):
            return str(path)

        import ctypes, ctypes.wintypes as wt

        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [wt.LPCWSTR, wt.LPWSTR, wt.DWORD]

        # the file must exist, or Windows will not create a short name
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).touch(exist_ok=True)

        buf = ctypes.create_unicode_buffer(260)
        if GetShortPathNameW(str(path), buf, 260):
            return buf.value
        return str(path)

    def solve(self, model, **kwds):
        ####################################################################
        # Presolve
        ####################################################################
        # Begin time tracking
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)

        # Update configuration options, based on keywords passed to solve
        # preserve_implicit=True is required to extract solver_options ConfigDict
        config: GAMSConfig = self.config(value=kwds, preserve_implicit=True)

        # Check if solver is available
        avail = self.available()

        if not avail:
            raise ApplicationError(
                f'Solver {self.__class__} is not available ({avail}).'
            )

        if config.timer is None:
            timer = HierarchicalTimer()
        else:
            timer = config.timer
        StaleFlagManager.mark_all_as_stale()

        config.writer_config.setdefault(
            "put_results_format", 'gdx' if gdxcc_available else 'dat'
        )

        # local variable to hold the working directory name and flags
        dname = None
        lst = "output.lst"
        output_filename = None
        with TempfileManager.new_context() as tempfile:
            # IMPORTANT - only delete the whole tmpdir if the solver was the one
            # that made the directory. Otherwise, just delete the files the solver
            # made, if not keepfiles. That way the user can select a directory
            # they already have, like the current directory, without having to
            # worry about the rest of the contents of that directory being deleted.
            if config.working_dir is None:
                dname = tempfile.mkdtemp()
            else:
                dname = config.working_dir
            if not os.path.exists(dname):
                os.mkdir(dname)
            basename = os.path.join(dname, model.name)
            output_filename = basename + '.gms'
            lst_filename = os.path.join(dname, lst)
            with open(output_filename, 'w', newline='\n', encoding='utf-8') as gms_file:
                timer.start(f'write_{output_filename}_file')
                self._writer.config.set_value(config.writer_config)

                # update the writer config if any of the overlapping keys exists in the solver_options
                non_solver_config = {}
                for key in config.solver_options.keys():
                    if key in self._writer.config:
                        self._writer.config[key] = config.solver_options[key]
                    else:
                        non_solver_config[key] = config.solver_options[key]

                self._writer.config['add_options'] = non_solver_config

                gms_info = self._writer.write(model, gms_file, **self._writer.config)

                # NOTE: omit InfeasibleConstraintException for now
                timer.stop(f'write_{output_filename}_file')
            if config.writer_config.put_results_format == 'gdx':
                results_filename = os.path.join(dname, f"{model.name}_p.gdx")
                statresults_filename = os.path.join(
                    dname, "%s_s.gdx" % (config.writer_config.put_results,)
                )
            else:
                results_filename = os.path.join(
                    dname, "%s.dat" % (config.writer_config.put_results,)
                )
                statresults_filename = os.path.join(
                    dname, "%sstat.dat" % (config.writer_config.put_results,)
                )

            ####################################################################
            # Apply solver
            ####################################################################
            exe_path = config.executable.path()
            command = [exe_path, output_filename, "o=" + lst, "curdir=" + dname]

            if config.tee and not config.logfile:
                # default behaviour of gams is to print to console, for
                # compatibility with windows and *nix we want to explicitly log to
                # stdout (see https://www.gams.com/latest/docs/UG_GamsCall.html)
                command.append("lo=3")
            elif not config.tee and not config.logfile:
                command.append("lo=0")
            elif not config.tee and config.logfile:
                command.append("lo=2")
            elif config.tee and config.logfile:
                command.append("lo=4")
            if config.logfile:
                command.append(f"lf={self._rewrite_path_win8p3(config.logfile)}")
            ostreams = [StringIO()]
            if config.tee:
                ostreams.append(sys.stdout)
            with TeeStream(*ostreams) as t:
                timer.start('subprocess')
                subprocess_result = subprocess.run(
                    command, stdout=t.STDOUT, stderr=t.STDERR
                )
                timer.stop('subprocess')
            rc = subprocess_result.returncode
            txt = ostreams[0].getvalue()
            if config.working_dir:
                print("\nGAMS WORKING DIRECTORY: %s\n" % config.working_dir)

            if rc == 1 or rc == 127:
                raise IOError("Command 'gams' was not recognized")
            elif rc != 0:
                if rc == 3:
                    # Execution Error
                    # Run check_expr_evaluation, which errors if necessary
                    print('Error rc=3, to be determined later')
                # If nothing was raised, or for all other cases, raise this
                logger.error(
                    "GAMS encountered an error during solve. "
                    "Check listing file for details."
                )
                logger.error(txt)
                if os.path.exists(lst_filename):
                    with open(lst_filename, 'r') as FILE:
                        logger.error("GAMS Listing file:\n\n%s" % (FILE.read(),))
                raise RuntimeError(
                    "GAMS encountered an error during solve. "
                    "Check listing file for details."
                )
            if config.writer_config.put_results_format == 'gdx':
                timer.start('parse_gdx')
                model_soln, stat_vars = self._parse_gdx_results(
                    config, results_filename, statresults_filename
                )
                timer.stop('parse_gdx')

            else:
                timer.start('parse_dat')
                model_soln, stat_vars = self._parse_dat_results(
                    config, results_filename, statresults_filename
                )
                timer.stop('parse_dat')

            ####################################################################
            # Postsolve (WIP)
            ####################################################################

            # Mapping between old and new contrib results
            rev_legacy_termination_condition_map = {
                v: k for k, v in legacy_termination_condition_map.items()
            }

            model_suffixes = list(
                name
                for (
                    name,
                    comp,
                ) in pyomo.core.base.suffix.active_import_suffix_generator(model)
            )
            extract_dual = 'dual' in model_suffixes
            extract_rc = 'rc' in model_suffixes
            results = GAMSResults()
            results.solver_name = "GAMS "
            results.solver_version = self.version()

            solvestat = stat_vars["SOLVESTAT"]
            if solvestat == 1:
                results.gams_solver_status = SolverStatus.ok
            elif solvestat == 2:
                results.gams_solver_status = SolverStatus.ok
                results.gams_termination_condition = TerminationCondition.maxIterations
            elif solvestat == 3:
                results.gams_solver_status = SolverStatus.ok
                results.gams_termination_condition = TerminationCondition.maxTimeLimit
            elif solvestat == 5:
                results.gams_solver_status = SolverStatus.ok
                results.gams_termination_condition = TerminationCondition.maxEvaluations
            elif solvestat == 7:
                results.gams_solver_status = SolverStatus.aborted
                results.gams_termination_condition = (
                    TerminationCondition.licensingProblems
                )
            elif solvestat == 8:
                results.gams_solver_status = SolverStatus.aborted
                results.gams_termination_condition = TerminationCondition.userInterrupt
            elif solvestat == 10:
                results.gams_solver_status = SolverStatus.error
                results.gams_termination_condition = TerminationCondition.solverFailure
            elif solvestat == 11:
                results.gams_solver_status = SolverStatus.error
                results.gams_termination_condition = (
                    TerminationCondition.internalSolverError
                )
            elif solvestat == 4:
                results.gams_solver_status = SolverStatus.warning
                results.message = "Solver quit with a problem (see LST file)"
            elif solvestat in (9, 12, 13):
                results.gams_solver_status = SolverStatus.error
            elif solvestat == 6:
                results.gams_solver_status = SolverStatus.unknown

            modelstat = stat_vars["MODELSTAT"]
            if modelstat == 1:
                results.gams_termination_condition = TerminationCondition.optimal
                results.solution_status = SolutionStatus.optimal
            elif modelstat == 2:
                results.gams_termination_condition = TerminationCondition.locallyOptimal
                results.solution_status = SolutionStatus.feasible
            elif modelstat in [3, 18]:
                results.gams_termination_condition = TerminationCondition.unbounded
                # results.solution_status = SolutionStatus.unbounded
                results.solution_status = SolutionStatus.noSolution

            elif modelstat in [4, 5, 6, 10, 19]:
                results.gams_termination_condition = (
                    TerminationCondition.infeasibleOrUnbounded
                )
                results.solution_status = SolutionStatus.infeasible
                raise InfeasibleConstraintException('Solver status returns infeasible')
            elif modelstat == 7:
                results.gams_termination_condition = TerminationCondition.feasible
                results.solution_status = SolutionStatus.feasible
            elif modelstat == 8:
                # 'Integer solution model found'
                results.gams_termination_condition = TerminationCondition.optimal
                results.solution_status = SolutionStatus.optimal
            elif modelstat == 9:
                results.gams_termination_condition = (
                    TerminationCondition.intermediateNonInteger
                )
                results.solution_status = SolutionStatus.noSolution
            elif modelstat == 11:
                # Should be handled above, if modelstat and solvestat both
                # indicate a licensing problem
                if results.gams_termination_condition is None:
                    results.gams_termination_condition = (
                        TerminationCondition.licensingProblems
                    )
                results.solution_status = SolutionStatus.noSolution
                # results.solution_status = SolutionStatus.error

            elif modelstat in [12, 13]:
                if results.gams_termination_condition is None:
                    results.gams_termination_condition = TerminationCondition.error
                results.solution_status = SolutionStatus.noSolution
                # results.solution_status = SolutionStatus.error

            elif modelstat == 14:
                if results.gams_termination_condition is None:
                    results.gams_termination_condition = TerminationCondition.noSolution
                results.solution_status = SolutionStatus.noSolution
                # results.solution_status = SolutionStatus.unknown

            elif modelstat in [15, 16, 17]:
                # Having to do with CNS models,
                # not sure what to make of status descriptions
                results.gams_termination_condition = TerminationCondition.optimal
                results.solution_status = SolutionStatus.noSolution
            else:
                # This is just a backup catch, all cases are handled above
                results.solution_status = SolutionStatus.noSolution

            # ensure backward compatibility before feeding to contrib.solver
            results.termination_condition = rev_legacy_termination_condition_map[
                results.gams_termination_condition
            ]
            obj = list(model.component_data_objects(Objective, active=True))

            # NOTE: How should gams handle when no objective is provided
            # NOTE: pyomo/contrib/solver/tests/solvers/test_solvers.py::TestSolvers::test_no_objective
            # NOTE: results.incumbent_objective = None
            # NOTE: results.objective_bound = None
            # assert len(obj) == 1, 'Only one objective is allowed.'

            if results.solution_status in {
                SolutionStatus.feasible,
                SolutionStatus.optimal,
            }:
                results.solution_loader = GMSSolutionLoader(
                    gdx_data=model_soln, gms_info=gms_info
                )

                if config.load_solutions:
                    results.solution_loader.load_vars()
                    if len(obj) == 1:
                        results.incumbent_objective = stat_vars["OBJVAL"]
                    else:
                        results.incumbent_objective = None
                    if (
                        hasattr(model, 'dual')
                        and isinstance(model.dual, Suffix)
                        and model.dual.import_enabled()
                    ):
                        model.dual.update(results.solution_loader.get_duals())
                    if (
                        hasattr(model, 'rc')
                        and isinstance(model.rc, Suffix)
                        and model.rc.import_enabled()
                    ):
                        model.rc.update(results.solution_loader.get_reduced_costs())

                else:
                    results.incumbent_objective = value(
                        replace_expressions(
                            obj[0].expr,
                            substitution_map={
                                id(v): val
                                for v, val in results.solution_loader.get_primals().items()
                            },
                            descend_into_named_expressions=True,
                            remove_named_expressions=True,
                        )
                    )
            end_timestamp = datetime.datetime.now(datetime.timezone.utc)
            results.timing_info.start_timestamp = start_timestamp
            results.timing_info.wall_time = (
                end_timestamp - start_timestamp
            ).total_seconds()
            results.timing_info.timer = timer
            return results

    def _parse_gdx_results(self, config, results_filename, statresults_filename):
        model_soln = dict()
        stat_vars = dict.fromkeys(
            [
                'MODELSTAT',
                'SOLVESTAT',
                'OBJEST',
                'OBJVAL',
                'NUMVAR',
                'NUMEQU',
                'NUMDVAR',
                'NUMNZ',
                'ETSOLVE',
            ]
        )

        pgdx = gdxcc.new_gdxHandle_tp()
        ret = gdxcc.gdxCreateD(pgdx, os.path.dirname(config.executable.path()), 128)
        if not ret[0]:
            raise RuntimeError("GAMS GDX failure (gdxCreate): %s." % ret[1])
        if os.path.exists(statresults_filename):
            ret = gdxcc.gdxOpenRead(pgdx, statresults_filename)
            if not ret[0]:
                raise RuntimeError("GAMS GDX failure (gdxOpenRead): %d." % ret[1])

            specVals = gdxcc.doubleArray(gdxcc.GMS_SVIDX_MAX)
            rc = gdxcc.gdxGetSpecialValues(pgdx, specVals)

            specVals[gdxcc.GMS_SVIDX_EPS] = sys.float_info.min
            specVals[gdxcc.GMS_SVIDX_UNDEF] = float("nan")
            specVals[gdxcc.GMS_SVIDX_PINF] = float("inf")
            specVals[gdxcc.GMS_SVIDX_MINF] = float("-inf")
            specVals[gdxcc.GMS_SVIDX_NA] = struct.unpack(
                ">d", bytes.fromhex("fffffffffffffffe")
            )[0]
            gdxcc.gdxSetSpecialValues(pgdx, specVals)

            i = 0
            while True:
                i += 1
                ret = gdxcc.gdxDataReadRawStart(pgdx, i)
                if not ret[0]:
                    break

                ret = gdxcc.gdxSymbolInfo(pgdx, i)
                if not ret[0]:
                    break
                if len(ret) < 2:
                    raise RuntimeError("GAMS GDX failure (gdxSymbolInfo).")
                stat = ret[1]
                if not stat in stat_vars:
                    continue

                ret = gdxcc.gdxDataReadRaw(pgdx)
                if not ret[0] or len(ret[2]) == 0:
                    raise RuntimeError("GAMS GDX failure (gdxDataReadRaw).")

                if stat in ('OBJEST', 'OBJVAL', 'ETSOLVE'):
                    stat_vars[stat] = ret[2][0]
                else:
                    stat_vars[stat] = int(ret[2][0])

            gdxcc.gdxDataReadDone(pgdx)
            gdxcc.gdxClose(pgdx)

        if os.path.exists(results_filename):
            ret = gdxcc.gdxOpenRead(pgdx, results_filename)
            if not ret[0]:
                raise RuntimeError("GAMS GDX failure (gdxOpenRead): %d." % ret[1])

            specVals = gdxcc.doubleArray(gdxcc.GMS_SVIDX_MAX)
            rc = gdxcc.gdxGetSpecialValues(pgdx, specVals)

            specVals[gdxcc.GMS_SVIDX_EPS] = sys.float_info.min
            specVals[gdxcc.GMS_SVIDX_UNDEF] = float("nan")
            specVals[gdxcc.GMS_SVIDX_PINF] = float("inf")
            specVals[gdxcc.GMS_SVIDX_MINF] = float("-inf")
            specVals[gdxcc.GMS_SVIDX_NA] = struct.unpack(
                ">d", bytes.fromhex("fffffffffffffffe")
            )[0]
            gdxcc.gdxSetSpecialValues(pgdx, specVals)

            i = 0
            while True:
                i += 1
                ret = gdxcc.gdxDataReadRawStart(pgdx, i)
                if not ret[0]:
                    break

                ret = gdxcc.gdxDataReadRaw(pgdx)
                if not ret[0] or len(ret[2]) < 2:
                    raise RuntimeError("GAMS GDX failure (gdxDataReadRaw).")
                level = ret[2][0]
                dual = ret[2][1]

                ret = gdxcc.gdxSymbolInfo(pgdx, i)
                if not ret[0]:
                    break
                if len(ret) < 2:
                    raise RuntimeError("GAMS GDX failure (gdxSymbolInfo).")
                model_soln[ret[1]] = (level, dual)

            gdxcc.gdxDataReadDone(pgdx)
            gdxcc.gdxClose(pgdx)

        gdxcc.gdxFree(pgdx)
        gdxcc.gdxLibraryUnload()
        return model_soln, stat_vars

    def _parse_dat_results(self, config, results_filename, statresults_filename):
        with open(statresults_filename, 'r') as statresults_file:
            statresults_text = statresults_file.read()

        stat_vars = dict()
        # Skip first line of explanatory text
        for line in statresults_text.splitlines()[1:]:
            items = line.split()
            try:
                stat_vars[items[0]] = float(items[1])
            except ValueError:
                # GAMS printed NA, just make it nan
                stat_vars[items[0]] = float('nan')

        with open(results_filename, 'r') as results_file:
            results_text = results_file.read()

        model_soln = dict()
        # Skip first line of explanatory text
        for line in results_text.splitlines()[1:]:
            items = line.split()
            model_soln[items[0]] = (float(items[1]), float(items[2]))

        return model_soln, stat_vars
