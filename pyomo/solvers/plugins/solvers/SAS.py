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
import sys
from os import stat
from abc import ABC, abstractmethod
from io import StringIO

from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import SolverFactory
from pyomo.common.collections import Bunch
from pyomo.common.dependencies import attempt_import
from pyomo.opt.results import (
    SolverResults,
    SolverStatus,
    TerminationCondition,
    SolutionStatus,
    ProblemSense,
)
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base import Var
from pyomo.core.base.block import BlockData
from pyomo.core.kernel.block import IBlock
from pyomo.common.log import LogStream
from pyomo.common.tee import capture_output, TeeStream


uuid, uuid_available = attempt_import('uuid')
logger = logging.getLogger("pyomo.solvers")


STATUS_TO_SOLVERSTATUS = {
    "OK": SolverStatus.ok,
    "SYNTAX_ERROR": SolverStatus.error,
    "DATA_ERROR": SolverStatus.error,
    "OUT_OF_MEMORY": SolverStatus.aborted,
    "IO_ERROR": SolverStatus.error,
    "ERROR": SolverStatus.error,
}

# This combines all status codes from OPTLP/solvelp and OPTMILP/solvemilp
SOLSTATUS_TO_TERMINATIONCOND = {
    "OPTIMAL": TerminationCondition.optimal,
    "OPTIMAL_AGAP": TerminationCondition.optimal,
    "OPTIMAL_RGAP": TerminationCondition.optimal,
    "OPTIMAL_COND": TerminationCondition.optimal,
    "TARGET": TerminationCondition.optimal,
    "CONDITIONAL_OPTIMAL": TerminationCondition.optimal,
    "FEASIBLE": TerminationCondition.feasible,
    "INFEASIBLE": TerminationCondition.infeasible,
    "UNBOUNDED": TerminationCondition.unbounded,
    "INFEASIBLE_OR_UNBOUNDED": TerminationCondition.infeasibleOrUnbounded,
    "SOLUTION_LIM": TerminationCondition.maxEvaluations,
    "NODE_LIM_SOL": TerminationCondition.maxEvaluations,
    "NODE_LIM_NOSOL": TerminationCondition.maxEvaluations,
    "ITERATION_LIMIT_REACHED": TerminationCondition.maxIterations,
    "TIME_LIM_SOL": TerminationCondition.maxTimeLimit,
    "TIME_LIM_NOSOL": TerminationCondition.maxTimeLimit,
    "TIME_LIMIT_REACHED": TerminationCondition.maxTimeLimit,
    "ABORTED": TerminationCondition.userInterrupt,
    "ABORT_SOL": TerminationCondition.userInterrupt,
    "ABORT_NOSOL": TerminationCondition.userInterrupt,
    "OUTMEM_SOL": TerminationCondition.solverFailure,
    "OUTMEM_NOSOL": TerminationCondition.solverFailure,
    "FAILED": TerminationCondition.solverFailure,
    "FAIL_SOL": TerminationCondition.solverFailure,
    "FAIL_NOSOL": TerminationCondition.solverFailure,
}


SOLSTATUS_TO_MESSAGE = {
    "OPTIMAL": "The solution is optimal.",
    "OPTIMAL_AGAP": "The solution is optimal within the absolute gap specified by the ABSOBJGAP= option.",
    "OPTIMAL_RGAP": "The solution is optimal within the relative gap specified by the RELOBJGAP= option.",
    "OPTIMAL_COND": "The solution is optimal, but some infeasibilities (primal, bound, or integer) exceed tolerances due to scaling or choice of a small INTTOL= value.",
    "TARGET": "The solution is not worse than the target specified by the TARGET= option.",
    "CONDITIONAL_OPTIMAL": "The solution is optimal, but some infeasibilities (primal, dual or bound) exceed tolerances due to scaling or preprocessing.",
    "FEASIBLE": "The problem is feasible. This status is displayed when the IIS=TRUE option is specified and the problem is feasible.",
    "INFEASIBLE": "The problem is infeasible.",
    "UNBOUNDED": "The problem is unbounded.",
    "INFEASIBLE_OR_UNBOUNDED": "The problem is infeasible or unbounded.",
    "SOLUTION_LIM": "The solver reached the maximum number of solutions specified by the MAXSOLS= option.",
    "NODE_LIM_SOL": "The solver reached the maximum number of nodes specified by the MAXNODES= option and found a solution.",
    "NODE_LIM_NOSOL": "The solver reached the maximum number of nodes specified by the MAXNODES= option and did not find a solution.",
    "ITERATION_LIMIT_REACHED": "The maximum allowable number of iterations was reached.",
    "TIME_LIM_SOL": "The solver reached the execution time limit specified by the MAXTIME= option and found a solution.",
    "TIME_LIM_NOSOL": "The solver reached the execution time limit specified by the MAXTIME= option and did not find a solution.",
    "TIME_LIMIT_REACHED": "The solver reached its execution time limit.",
    "ABORTED": "The solver was interrupted externally.",
    "ABORT_SOL": "The solver was stopped by the user but still found a solution.",
    "ABORT_NOSOL": "The solver was stopped by the user and did not find a solution.",
    "OUTMEM_SOL": "The solver ran out of memory but still found a solution.",
    "OUTMEM_NOSOL": "The solver ran out of memory and either did not find a solution or failed to output the solution due to insufficient memory.",
    "FAILED": "The solver failed to converge, possibly due to numerical issues.",
    "FAIL_SOL": "The solver stopped due to errors but still found a solution.",
    "FAIL_NOSOL": "The solver stopped due to errors and did not find a solution.",
}


@SolverFactory.register("sas", doc="The SAS LP/MIP solver")
class SAS(OptSolver):
    """The SAS optimization solver"""

    def __new__(cls, *args, **kwds):
        mode = kwds.pop("solver_io", None)
        if mode != None:
            return SolverFactory(mode, **kwds)
        else:
            # Choose solver factory automatically
            # based on what can be loaded.
            s = SolverFactory("_sas94", **kwds)
            if not s.available():
                s = SolverFactory("_sascas", **kwds)
            return s


class SASAbc(ABC, OptSolver):
    """Abstract base class for the SAS solver interfaces. Simply to avoid code duplication."""

    def __init__(self, **kwds):
        """Initialize the SAS solver interfaces."""
        kwds["type"] = "sas"
        super(SASAbc, self).__init__(**kwds)

        #
        # Set up valid problem formats and valid results for each
        # problem format
        #
        self._valid_problem_formats = [ProblemFormat.mps]
        self._valid_result_formats = {ProblemFormat.mps: [ResultsFormat.soln]}

        self._keepfiles = False
        self._capabilities.linear = True
        self._capabilities.integer = True

        super(SASAbc, self).set_problem_format(ProblemFormat.mps)

    def _presolve(self, *args, **kwds):
        """Set things up for the actual solve."""
        # create a context in the temporary file manager for
        # this plugin - is "pop"ed in the _postsolve method.
        TempfileManager.push()

        # Get the warmstart flag
        self.warmstart_flag = kwds.pop("warmstart", False)

        # Call parent presolve function
        super(SASAbc, self)._presolve(*args, **kwds)

        # Store the model, too bad this is not done in the base class
        for arg in args:
            if isinstance(arg, (BlockData, IBlock)):
                # Store the instance
                self._instance = arg
                self._vars = []
                for block in self._instance.block_data_objects(active=True):
                    for vardata in block.component_data_objects(
                        Var, active=True, descend_into=False
                    ):
                        self._vars.append(vardata)
                # Store the symbol map, we need this for example when writing the warmstart file
                if isinstance(self._instance, IBlock):
                    self._smap = getattr(self._instance, "._symbol_maps")[self._smap_id]
                else:
                    self._smap = self._instance.solutions.symbol_map[self._smap_id]

        # Create the primalin data
        if self.warmstart_flag:
            filename = self._warm_start_file_name = TempfileManager.create_tempfile(
                ".sol", text=True
            )
            smap = self._smap
            numWritten = 0
            with open(filename, "w") as file:
                file.write("_VAR_,_VALUE_\n")
                for var in self._vars:
                    if (var.value is not None) and (id(var) in smap.byObject):
                        name = smap.byObject[id(var)]
                        file.write(
                            "{name},{value}\n".format(name=name, value=var.value)
                        )
                        numWritten += 1
            if numWritten == 0:
                # No solution available, disable warmstart
                self.warmstart_flag = False

    def available(self, exception_flag=False):
        """True if the solver is available"""
        if not self._python_api_exists:
            return False
        return self.start_sas_session() is not None

    def _has_integer_variables(self):
        """True if the problem has integer variables."""
        for vardata in self._vars:
            if vardata.is_binary() or vardata.is_integer():
                return True
        return False

    def _create_results_from_status(self, status, solution_status):
        """Create a results object and set the status code and messages."""
        results = SolverResults()
        results.solver.name = "SAS"
        results.solver.status = STATUS_TO_SOLVERSTATUS[status]
        results.solver.hasSolution = False
        if results.solver.status == SolverStatus.ok:
            results.solver.termination_condition = SOLSTATUS_TO_TERMINATIONCOND[
                solution_status
            ]
            results.solver.message = results.solver.termination_message = (
                SOLSTATUS_TO_MESSAGE[solution_status]
            )
            results.solver.status = TerminationCondition.to_solver_status(
                results.solver.termination_condition
            )
            if "OPTIMAL" in solution_status or "_SOL" in solution_status:
                results.solver.hasSolution = True
        elif results.solver.status == SolverStatus.aborted:
            results.solver.termination_condition = TerminationCondition.userInterrupt
            if solution_status != "ERROR":
                results.solver.message = results.solver.termination_message = (
                    SOLSTATUS_TO_MESSAGE[solution_status]
                )
        else:
            results.solver.termination_condition = TerminationCondition.error
            results.solver.message = results.solver.termination_message = (
                SOLSTATUS_TO_MESSAGE["FAILED"]
            )
        return results

    @abstractmethod
    def _apply_solver(self):
        pass

    def _postsolve(self):
        """Clean up at the end, especially the temp files."""
        # Let the base class deal with returning results.
        results = super(SASAbc, self)._postsolve()

        # Finally, clean any temporary files registered with the temp file
        # manager, created populated *directly* by this plugin. does not
        # include, for example, the execution script. but does include
        # the warm-start file.
        TempfileManager.pop(remove=not self._keepfiles)

        return results

    def warm_start_capable(self):
        """True if the solver interface supports MILP warmstarting."""
        return True


@SolverFactory.register("_sas94", doc="SAS 9.4 interface")
class SAS94(SASAbc):
    """
    Solver interface for SAS 9.4 using saspy. See the saspy documentation about
    how to create a connection.
    The swat connection options can be specified on the SolverFactory call.
    """

    def __init__(self, **kwds):
        """Initialize the solver interface and see if the saspy package is available."""
        super(SAS94, self).__init__(**kwds)

        try:
            import saspy

            self._sas = saspy
        except ImportError:
            self._python_api_exists = False
        except Exception as e:
            self._python_api_exists = False
            # For other exceptions, raise it so that it does not get lost
            raise e
        else:
            self._python_api_exists = True
            self._sas.logger.setLevel(logger.level)

        # Store other options for the SAS session
        self._session_options = kwds
        self._sas_session = None

    def __del__(self):
        # Close the session, if we created one
        if self._sas_session:
            self._sas_session.endsas()
            del self._sas_session

    def _create_statement_str(self, statement):
        """Helper function to create the strings for the statements of the proc OPTLP/OPTMILP code."""
        stmt = self.options.pop(statement, None)
        if stmt:
            return (
                statement.strip()
                + " "
                + " ".join(option + "=" + str(value) for option, value in stmt.items())
                + ";"
            )
        else:
            return ""

    def sas_version(self):
        return self._sasver

    def start_sas_session(self):
        if self._sas_session is None:
            # Create (and cache) the session
            try:
                self._sas_session = self._sas.SASsession(**self._session_options)
            except:
                pass
        return self._sas_session

    def _apply_solver(self):
        """ "Prepare the options and run the solver. Then store the data to be returned."""
        logger.debug("Running SAS")

        # Set return code to issue an error if we get interrupted
        self._rc = -1

        # Figure out if the problem has integer variables
        with_opt = self.options.pop("with", None)
        if with_opt == "lp":
            proc = "OPTLP"
        elif with_opt == "milp":
            proc = "OPTMILP"
        else:
            # Check if there are integer variables, this might be slow
            proc = "OPTMILP" if self._has_integer_variables() else "OPTLP"

        # Get the rootnode options
        decomp_str = self._create_statement_str("decomp")
        decompmaster_str = self._create_statement_str("decompmaster")
        decompmasterip_str = self._create_statement_str("decompmasterip")
        decompsubprob_str = self._create_statement_str("decompsubprob")
        rootnode_str = self._create_statement_str("rootnode")

        # Get a unique identifier, always use the same with different prefixes
        unique = uuid.uuid4().hex[:16]

        # Create unique filename for output datasets
        primalout_dataset_name = "pout" + unique
        dualout_dataset_name = "dout" + unique
        primalin_dataset_name = None

        # Handle warmstart
        warmstart_str = ""
        if self.warmstart_flag:
            # Set the warmstart basis option
            primalin_dataset_name = "pin" + unique
            if proc != "OPTLP":
                warmstart_str = """
                                proc import datafile='{primalin}'
                                    out={primalin_dataset_name}
                                    dbms=csv
                                    replace;
                                    getnames=yes;
                                    run;
                                """.format(
                    primalin=self._warm_start_file_name,
                    primalin_dataset_name=primalin_dataset_name,
                )
                self.options["primalin"] = primalin_dataset_name

        # Convert options to string
        opt_str = " ".join(
            option + "=" + str(value) for option, value in self.options.items()
        )

        # Set some SAS options to make the log more clean
        sas_options = "option notes nonumber nodate nosource pagesize=max;"

        # Get the current SAS session, submit the code and return the results
        sas = self.start_sas_session()

        # Find the version of 9.4 we are using
        self._sasver = sas.sasver

        # Upload files, only if not accessible locally
        upload_mps = False
        if not sas.file_info(self._problem_files[0], quiet=True):
            sas.upload(self._problem_files[0], self._problem_files[0], overwrite=True)
            upload_mps = True

        upload_pin = False
        if self.warmstart_flag and not sas.file_info(
            self._warm_start_file_name, quiet=True
        ):
            sas.upload(
                self._warm_start_file_name, self._warm_start_file_name, overwrite=True
            )
            upload_pin = True

        # Using a function call to make it easier to mock the version check
        major_version = self.sas_version()[0]
        minor_version = self.sas_version().split("M", 1)[1][0]
        if major_version == "9" and int(minor_version) < 5:
            raise NotImplementedError(
                "Support for SAS 9.4 M4 and earlier is not implemented."
            )
        elif major_version == "9" and int(minor_version) == 5:
            # In 9.4M5 we have to create an MPS data set from an MPS file first
            # Earlier versions will not work because the MPS format in incompatible
            mps_dataset_name = "mps" + unique
            res = sas.submit(
                """
                            {sas_options}
                            {warmstart}
                            %MPS2SASD(MPSFILE="{mpsfile}", OUTDATA={mps_dataset_name}, MAXLEN=256, FORMAT=FREE);
                            proc {proc} data={mps_dataset_name} {options} primalout={primalout_dataset_name} dualout={dualout_dataset_name};
                            {decomp}
                            {decompmaster}
                            {decompmasterip}
                            {decompsubprob}
                            {rootnode}
                            run;
                            """.format(
                    sas_options=sas_options,
                    warmstart=warmstart_str,
                    proc=proc,
                    mpsfile=self._problem_files[0],
                    mps_dataset_name=mps_dataset_name,
                    options=opt_str,
                    primalout_dataset_name=primalout_dataset_name,
                    dualout_dataset_name=dualout_dataset_name,
                    decomp=decomp_str,
                    decompmaster=decompmaster_str,
                    decompmasterip=decompmasterip_str,
                    decompsubprob=decompsubprob_str,
                    rootnode=rootnode_str,
                ),
                results="TEXT",
            )
            sas.sasdata(mps_dataset_name).delete(quiet=True)
        else:
            # Since 9.4M6+ optlp/optmilp can read mps files directly (this includes Viya-based local installs)
            res = sas.submit(
                """
                            {sas_options}
                            {warmstart}
                            proc {proc} mpsfile=\"{mpsfile}\" {options} primalout={primalout_dataset_name} dualout={dualout_dataset_name};
                            {decomp}
                            {decompmaster}
                            {decompmasterip}
                            {decompsubprob}
                            {rootnode}
                            run;
                            """.format(
                    sas_options=sas_options,
                    warmstart=warmstart_str,
                    proc=proc,
                    mpsfile=self._problem_files[0],
                    options=opt_str,
                    primalout_dataset_name=primalout_dataset_name,
                    dualout_dataset_name=dualout_dataset_name,
                    decomp=decomp_str,
                    decompmaster=decompmaster_str,
                    decompmasterip=decompmasterip_str,
                    decompsubprob=decompsubprob_str,
                    rootnode=rootnode_str,
                ),
                results="TEXT",
            )

        # Delete uploaded file
        if upload_mps:
            sas.file_delete(self._problem_files[0], quiet=True)
        if self.warmstart_flag and upload_pin:
            sas.file_delete(self._warm_start_file_name, quiet=True)

        # Store log and ODS output
        self._log = res["LOG"]
        self._lst = res["LST"]
        if "ERROR 22-322: Syntax error" in self._log:
            raise ValueError(
                "An option passed to the SAS solver caused a syntax error: {log}".format(
                    log=self._log
                )
            )
        else:
            # Print log if requested by the user, only if we did not already print it
            if self._tee:
                print(self._log)
        self._macro = dict(
            (key.strip(), value.strip())
            for key, value in (
                pair.split("=") for pair in sas.symget("_OR" + proc + "_").split()
            )
        )
        if self._macro.get("STATUS", "ERROR") == "OK":
            primal_out = sas.sd2df(primalout_dataset_name)
            dual_out = sas.sd2df(dualout_dataset_name)

        # Delete data sets, they will go away automatically, but does not hurt to delete them
        if primalin_dataset_name:
            sas.sasdata(primalin_dataset_name).delete(quiet=True)
        sas.sasdata(primalout_dataset_name).delete(quiet=True)
        sas.sasdata(dualout_dataset_name).delete(quiet=True)

        # Prepare the solver results
        results = self.results = self._create_results_from_status(
            self._macro.get("STATUS", "ERROR"),
            self._macro.get("SOLUTION_STATUS", "ERROR"),
        )

        if "Objective Sense            Maximization" in self._lst:
            results.problem.sense = ProblemSense.maximize
        else:
            results.problem.sense = ProblemSense.minimize

        # Prepare the solution information
        if results.solver.hasSolution:
            sol = results.solution.add()

            # Store status in solution
            sol.status = SolutionStatus.feasible
            sol.termination_condition = SOLSTATUS_TO_TERMINATIONCOND[
                self._macro.get("SOLUTION_STATUS", "ERROR")
            ]

            # Store objective value in solution
            sol.objective["__default_objective__"] = {"Value": self._macro["OBJECTIVE"]}

            if proc == "OPTLP":
                # Convert primal out data set to variable dictionary
                # Use pandas functions for efficiency
                primal_out = primal_out[["_VAR_", "_VALUE_", "_STATUS_", "_R_COST_"]]
                primal_out = primal_out.set_index("_VAR_", drop=True)
                primal_out = primal_out.rename(
                    {"_VALUE_": "Value", "_STATUS_": "Status", "_R_COST_": "rc"},
                    axis="columns",
                )
                sol.variable = primal_out.to_dict("index")

                # Convert dual out data set to constraint dictionary
                # Use pandas functions for efficiency
                dual_out = dual_out[["_ROW_", "_VALUE_", "_STATUS_", "_ACTIVITY_"]]
                dual_out = dual_out.set_index("_ROW_", drop=True)
                dual_out = dual_out.rename(
                    {"_VALUE_": "dual", "_STATUS_": "Status", "_ACTIVITY_": "slack"},
                    axis="columns",
                )
                sol.constraint = dual_out.to_dict("index")
            else:
                # Convert primal out data set to variable dictionary
                # Use pandas functions for efficiency
                primal_out = primal_out[["_VAR_", "_VALUE_"]]
                primal_out = primal_out.set_index("_VAR_", drop=True)
                primal_out = primal_out.rename({"_VALUE_": "Value"}, axis="columns")
                sol.variable = primal_out.to_dict("index")

        self._rc = 0
        return Bunch(rc=self._rc, log=self._log)


@SolverFactory.register("_sascas", doc="SAS Viya CAS Server interface")
class SASCAS(SASAbc):
    """
    Solver interface connection to a SAS Viya CAS server using swat.
    See the documentation for the swat package about how to create a connection.
    The swat connection options can be specified on the SolverFactory call.
    """

    def __init__(self, **kwds):
        """Initialize and try to load the swat package."""
        super(SASCAS, self).__init__(**kwds)

        try:
            import swat

            self._sas = swat
        except ImportError:
            self._python_api_exists = False
        except Exception as e:
            self._python_api_exists = False
            # For other exceptions, raise it so that it does not get lost
            raise e
        else:
            self._python_api_exists = True

        self._session_options = kwds
        self._sas_session = None

    def __del__(self):
        # Close the session, if we created one
        if self._sas_session:
            self._sas_session.close()
            del self._sas_session

    def start_sas_session(self):
        if self._sas_session is None:
            # Create (and cache) the session
            try:
                self._sas_session = self._sas.CAS(**self._session_options)
            except:
                pass
        return self._sas_session

    def _uploadMpsFile(self, s, unique):
        # Declare a unique table name for the mps table
        mpsdata_table_name = "mps" + unique

        # Upload mps file to CAS, if the file is larger than 2 GB, we need to use convertMps instead of loadMps
        # Note that technically it is 2 Gibibytes file size that trigger the issue, but 2 GB is the safer threshold
        if stat(self._problem_files[0]).st_size > 2e9:
            # For files larger than 2 GB (this is a limitation of the loadMps action used in the else part).
            # Use convertMPS, first create file for upload.
            mpsWithIdFileName = TempfileManager.create_tempfile(".mps.csv", text=True)
            with open(mpsWithIdFileName, "w") as mpsWithId:
                mpsWithId.write("_ID_\tText\n")
                with open(self._problem_files[0], "r") as f:
                    id = 0
                    for line in f:
                        id += 1
                        mpsWithId.write(str(id) + "\t" + line.rstrip() + "\n")

            # Upload .mps.csv file
            mpscsv_table_name = "csv" + unique
            s.upload_file(
                mpsWithIdFileName,
                casout={"name": mpscsv_table_name, "replace": True},
                importoptions={"filetype": "CSV", "delimiter": "\t"},
            )

            # Convert .mps.csv file to .mps
            s.optimization.convertMps(
                data=mpscsv_table_name,
                casOut={"name": mpsdata_table_name, "replace": True},
                format="FREE",
                maxLength=256,
            )

            # Delete the table we don't need anymore
            if mpscsv_table_name:
                s.dropTable(name=mpscsv_table_name, quiet=True)
        else:
            # For small files (less than 2 GB), use loadMps
            with open(self._problem_files[0], "r") as mps_file:
                s.optimization.loadMps(
                    mpsFileString=mps_file.read(),
                    casout={"name": mpsdata_table_name, "replace": True},
                    format="FREE",
                    maxLength=256,
                )
        return mpsdata_table_name

    def _uploadPrimalin(self, s, unique):
        # Upload warmstart file to CAS with a unique name
        primalin_table_name = "pin" + unique
        s.upload_file(
            self._warm_start_file_name,
            casout={"name": primalin_table_name, "replace": True},
            importoptions={"filetype": "CSV"},
        )
        self.options["primalin"] = primalin_table_name
        return primalin_table_name

    def _retrieveSolution(
        self, s, r, results, action, primalout_table_name, dualout_table_name
    ):
        # Create solution
        sol = results.solution.add()

        # Store status in solution
        sol.status = SolutionStatus.feasible
        sol.termination_condition = SOLSTATUS_TO_TERMINATIONCOND[
            r.get("solutionStatus", "ERROR")
        ]

        # Store objective value in solution
        sol.objective["__default_objective__"] = {"Value": r["objective"]}

        if action == "solveMilp":
            primal_out = s.CASTable(name=primalout_table_name)
            # Use pandas functions for efficiency
            primal_out = primal_out[["_VAR_", "_VALUE_"]]
            sol.variable = {}
            for row in primal_out.itertuples(index=False):
                sol.variable[row[0]] = {"Value": row[1]}
        else:
            # Convert primal out data set to variable dictionary
            # Use panda functions for efficiency
            primal_out = s.CASTable(name=primalout_table_name)
            primal_out = primal_out[["_VAR_", "_VALUE_", "_STATUS_", "_R_COST_"]]
            sol.variable = {}
            for row in primal_out.itertuples(index=False):
                sol.variable[row[0]] = {"Value": row[1], "Status": row[2], "rc": row[3]}

            # Convert dual out data set to constraint dictionary
            # Use pandas functions for efficiency
            dual_out = s.CASTable(name=dualout_table_name)
            dual_out = dual_out[["_ROW_", "_VALUE_", "_STATUS_", "_ACTIVITY_"]]
            sol.constraint = {}
            for row in dual_out.itertuples(index=False):
                sol.constraint[row[0]] = {
                    "dual": row[1],
                    "Status": row[2],
                    "slack": row[3],
                }

    def _apply_solver(self):
        """ "Prepare the options and run the solver. Then store the data to be returned."""
        logger.debug("Running SAS Viya")

        # Set return code to issue an error if we get interrupted
        self._rc = -1

        # Figure out if the problem has integer variables
        with_opt = self.options.pop("with", None)
        if with_opt == "lp":
            action = "solveLp"
        elif with_opt == "milp":
            action = "solveMilp"
        else:
            # Check if there are integer variables, this might be slow
            action = "solveMilp" if self._has_integer_variables() else "solveLp"

        # Get a unique identifier, always use the same with different prefixes
        unique = uuid.uuid4().hex[:16]

        # Creat the output stream, we want to print to a log string as well as to the console
        self._log = StringIO()
        ostreams = [LogStream(level=logging.INFO, logger=logger)]
        ostreams.append(self._log)
        if self._tee:
            ostreams.append(sys.stdout)

        # Connect to CAS server
        with capture_output(output=TeeStream(*ostreams), capture_fd=False):
            s = self.start_sas_session()
            try:
                # Load the optimization action set
                s.loadactionset("optimization")

                mpsdata_table_name = self._uploadMpsFile(s, unique)

                primalin_table_name = None
                if self.warmstart_flag:
                    primalin_table_name = self._uploadPrimalin(s, unique)

                # Define output table names
                primalout_table_name = "pout" + unique
                dualout_table_name = None

                # Solve the problem in CAS
                if action == "solveMilp":
                    r = s.optimization.solveMilp(
                        data={"name": mpsdata_table_name},
                        primalOut={"name": primalout_table_name, "replace": True},
                        **self.options,
                    )
                else:
                    dualout_table_name = "dout" + unique
                    r = s.optimization.solveLp(
                        data={"name": mpsdata_table_name},
                        primalOut={"name": primalout_table_name, "replace": True},
                        dualOut={"name": dualout_table_name, "replace": True},
                        **self.options,
                    )

                # Prepare the solver results
                if r:
                    # Get back the primal and dual solution data sets
                    results = self.results = self._create_results_from_status(
                        r.get("status", "ERROR"), r.get("solutionStatus", "ERROR")
                    )

                    if results.solver.status != SolverStatus.error:
                        if r.ProblemSummary["cValue1"][1] == "Maximization":
                            results.problem.sense = ProblemSense.maximize
                        else:
                            results.problem.sense = ProblemSense.minimize

                        # Prepare the solution information
                        if results.solver.hasSolution:
                            self._retrieveSolution(
                                s,
                                r,
                                results,
                                action,
                                primalout_table_name,
                                dualout_table_name,
                            )
                    else:
                        raise ValueError("The SAS solver returned an error status.")
                else:
                    results = self.results = SolverResults()
                    results.solver.name = "SAS"
                    results.solver.status = SolverStatus.error
                    raise ValueError(
                        "An option passed to the SAS solver caused a syntax error."
                    )

            finally:
                if mpsdata_table_name:
                    s.dropTable(name=mpsdata_table_name, quiet=True)
                if primalin_table_name:
                    s.dropTable(name=primalin_table_name, quiet=True)
                if primalout_table_name:
                    s.dropTable(name=primalout_table_name, quiet=True)
                if dualout_table_name:
                    s.dropTable(name=dualout_table_name, quiet=True)

        self._log = self._log.getvalue()
        self._rc = 0
        return Bunch(rc=self._rc, log=self._log)
