# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
import os
import subprocess
import time
import datetime
from io import StringIO
import sys
import struct
import re
import pathlib

from pyomo.common.dependencies import attempt_import
from pyomo.common.fileutils import Executable, ExecutableData
from pyomo.common.config import (
    ConfigValue,
    ConfigDict,
    document_configdict,
    Path,
    document_class_CONFIG,
)
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base import value, Objective
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.solver.common.base import SolverBase, Availability
from pyomo.contrib.solver.common.config import SolverConfig
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.solvers.gms_sol_reader import GMSSolutionLoader

import pyomo.core.base.suffix
from pyomo.common.tee import TeeStream
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.base.suffix import Suffix
from pyomo.common.errors import ApplicationError
from pyomo.contrib.solver.common.util import NoOptimalSolutionError
from pyomo.repn.plugins.gams_writer_v2 import GAMSWriter

logger = logging.getLogger(__name__)


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


@document_class_CONFIG(methods=['solve'])
class GAMS(SolverBase):
    CONFIG = GAMSConfig()

    # default behaviour of gams is to print to console, for
    # compatibility with windows and *nix we want to explicitly log to
    # stdout (see https://www.gams.com/latest/docs/UG_GamsCall.html)
    _log_levels = {
        (True, False): "lo=3",
        (False, False): "lo=0",
        (False, True): "lo=2",
        (True, True): "lo=4",
    }

    #: cache of availability / version information
    _exe_cache: dict = {}

    # mapping for GAMS SOLVESTAT
    _SOLVER_STATUS_LOOKUP = {
        1: None,
        2: TerminationCondition.iterationLimit,
        3: TerminationCondition.maxTimeLimit,
        4: TerminationCondition.error,
        5: TerminationCondition.iterationLimit,
        6: TerminationCondition.error,
        7: TerminationCondition.licensingProblems,
        8: TerminationCondition.interrupted,
        9: TerminationCondition.error,
        10: TerminationCondition.error,
        11: TerminationCondition.error,
        12: TerminationCondition.error,
        13: TerminationCondition.error,
    }

    # mapping for GAMS MODELSTAT
    _MODEL_STATUS_LOOKUP = {
        1: (SolutionStatus.optimal, TerminationCondition.convergenceCriteriaSatisfied),
        2: (SolutionStatus.feasible, TerminationCondition.convergenceCriteriaSatisfied),
        3: (SolutionStatus.noSolution, TerminationCondition.unbounded),
        4: (SolutionStatus.infeasible, TerminationCondition.provenInfeasible),
        5: (SolutionStatus.infeasible, TerminationCondition.locallyInfeasible),
        6: (SolutionStatus.infeasible, TerminationCondition.unknown),
        7: (SolutionStatus.feasible, TerminationCondition.unknown),
        8: (SolutionStatus.feasible, TerminationCondition.unknown),
        9: (SolutionStatus.noSolution, TerminationCondition.unknown),
        10: (SolutionStatus.infeasible, TerminationCondition.infeasibleOrUnbounded),
        11: (SolutionStatus.noSolution, TerminationCondition.licensingProblems),
        12: (SolutionStatus.noSolution, TerminationCondition.unknown),
        13: (SolutionStatus.noSolution, TerminationCondition.error),
        14: (SolutionStatus.noSolution, TerminationCondition.error),
        15: (SolutionStatus.optimal, TerminationCondition.convergenceCriteriaSatisfied),
        16: (
            SolutionStatus.feasible,
            TerminationCondition.convergenceCriteriaSatisfied,
        ),
        17: (
            SolutionStatus.feasible,
            TerminationCondition.convergenceCriteriaSatisfied,
        ),
        18: (SolutionStatus.noSolution, TerminationCondition.unbounded),
        19: (SolutionStatus.infeasible, TerminationCondition.provenInfeasible),
    }

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._writer = GAMSWriter()

    def available(self) -> Availability:
        ver, avail = self._get_version(self.config.executable.path())
        return avail

    def version(self):
        ver, avail = self._get_version(self.config.executable.path())
        return ver

    def _get_version(self, exe):
        # try cache
        try:
            return self._exe_cache[exe]
        except KeyError:
            pass

        if exe is None:
            exe = ""

        # exe must exist and must be executable, otherwise:
        if not (pathlib.Path(exe).exists() and os.access(exe, os.X_OK)):
            params = (None, Availability.NotFound)
            self._exe_cache[None] = params
            return params

        res = subprocess.run(
            [str(exe)], capture_output=True, encoding="utf8", check=False
        )
        if res.returncode:
            params = (None, Availability.NotFound)
            self._exe_cache[exe] = params
            logger.warning(
                f"Failed running GAMS command to get version (non-zero returncode):\n\n{res.stdout}"
            )
            return params

        version_pattern = r"(\d{1,3}\.\d{1,3}\.\d{1,3})"
        found = re.search(version_pattern, res.stdout)

        if not found:
            params = (None, Availability.NotFound)
            self._exe_cache[exe] = (None, Availability.NotFound)
            logger.warning(
                f"Failed parsing GAMS version (version not found while parsing):\n\n{res.stdout}"
            )
            return params

        version = found.group(0)
        version = tuple(int(i) for i in version.split('.'))

        if "Demo" in res.stdout:
            avail = Availability.LimitedLicense
        else:
            avail = Availability.FullLicense

        params = (version, avail)
        self._exe_cache[exe] = params
        return params

    def solve(self, model, **kwds):
        ####################################################################
        # Presolve
        ####################################################################
        # Begin time tracking
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        tick = time.perf_counter()

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
            config.timer = HierarchicalTimer()
        timer = config.timer
        StaleFlagManager.mark_all_as_stale()

        # local variable to hold the working directory name and flags
        dname = None
        lst = "output.lst"
        model_name = "model"

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
            basename = os.path.join(dname, model_name)
            output_filename = basename + '.gms'
            lst_filename = os.path.join(dname, lst)
            with open(output_filename, 'w', newline='\n', encoding='utf-8') as gms_file:
                timer.start(f'write_{output_filename}_file')
                self._writer.config.set_value(config.writer_config)
                self._writer.config.put_results_format = (
                    'gdx' if gdxcc_available else 'dat'
                )

                # update the writer config if any of the overlapping keys exists in the solver_options
                if config.time_limit is not None:
                    config.solver_options['resLim'] = config.time_limit

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
            if self._writer.config.put_results_format == 'gdx':
                results_filename = os.path.join(dname, "GAMS_MODEL_p.gdx")
                statresults_filename = os.path.join(
                    dname, "%s_s.gdx" % (self._writer.config.put_results,)
                )
            else:
                results_filename = os.path.join(
                    dname, "%s.dat" % (self._writer.config.put_results,)
                )
                statresults_filename = os.path.join(
                    dname, "%sstat.dat" % (self._writer.config.put_results,)
                )

            ####################################################################
            # Apply solver
            ####################################################################
            exe_path = config.executable.path()
            command = [exe_path, output_filename, "o=" + lst, "curdir=" + dname]

            # handled tee and logfile based on the length of list and string respectively
            command.append(self._log_levels[(bool(config.tee), bool(config.logfile))])

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
                logger.info("\nGAMS WORKING DIRECTORY: %s\n" % config.working_dir)

            if rc == 1 or rc == 127:
                raise IOError("Command 'gams' was not recognized")
            elif rc != 0:
                if rc == 3:
                    # Execution Error
                    # Run check_expr_evaluation, which errors if necessary
                    logger.error(
                        'Error rc=3 (GAMS execution error), to be determined later'
                    )
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
            if self._writer.config.put_results_format == 'gdx':
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
            results = self._postsolve(
                model, timer, config, model_soln, stat_vars, gms_info
            )

            results.solver_config = config
            results.solver_log = ostreams[0].getvalue()

            tock = time.perf_counter()
            results.timing_info.start_timestamp = start_timestamp
            results.timing_info.wall_time = tock - tick
            results.timing_info.timer = timer
            return results

    def _postsolve(self, model, timer, config, model_soln, stat_vars, gms_info):
        model_suffixes = list(
            name
            for (name, comp) in pyomo.core.base.suffix.active_import_suffix_generator(
                model
            )
        )
        extract_dual = 'dual' in model_suffixes
        extract_rc = 'rc' in model_suffixes
        results = Results()
        results.solver_name = "GAMS "
        results.solver_version = self.version()

        # --- Process GAMS SOLVESTAT ---
        solvestat = stat_vars["SOLVESTAT"]
        solver_term = GAMS._SOLVER_STATUS_LOOKUP.get(
            solvestat, TerminationCondition.unknown
        )

        # specific message for status 4
        if solvestat == 4:
            results.message = "Solver quit with a problem (see LST file)"

        # --- Process GAMS MODELSTAT ---
        modelstat = stat_vars["MODELSTAT"]
        solution_status, model_term = GAMS._MODEL_STATUS_LOOKUP.get(
            modelstat, (SolutionStatus.noSolution, TerminationCondition.unknown)
        )

        # --- Populate 'results'
        results.solution_status = solution_status

        # replaced below, if solution should be loaded
        results.solution_loader = GMSSolutionLoader(None, None)

        if solvestat == 1:
            results.termination_condition = model_term
        else:
            results.termination_condition = solver_term

        # extra_info (save GAMS status codes)
        results.extra_info.gams_solvestat = solvestat
        results.extra_info.gams_modelstat = modelstat

        # Taken from ipopt.py
        if (
            config.raise_exception_on_nonoptimal_result
            and results.solution_status != SolutionStatus.optimal
        ):
            raise NoOptimalSolutionError()

        obj = list(model.component_data_objects(Objective, active=True))

        if results.solution_status in {SolutionStatus.feasible, SolutionStatus.optimal}:
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
