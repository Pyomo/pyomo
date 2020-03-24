#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import logging
import sys
import time
import itertools
import math

import pyutilib.misc
from pyutilib.pyro import shutdown_pyro_components

import pyomo.solvers
from pyomo.core.base import ComponentUID
from pyomo.opt import (SolverFactory,
                       TerminationCondition,
                       UndefinedData,
                       ProblemFormat,
                       UnknownSolver,
                       SolutionStatus)
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    safe_declare_common_option,
                                    safe_declare_unique_option,
                                    _domain_percent,
                                    _domain_nonnegative,
                                    _domain_nonnegative_integer,
                                    _domain_positive_integer,
                                    _domain_must_be_str,
                                    _domain_unit_interval,
                                    _domain_tuple_of_str,
                                    _domain_tuple_of_str_or_dict,
                                    _output_options_group_title,
                                    _extension_options_group_title,
                                    _deprecated_options_group_title)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command,
                                  sort_extensions_by_precedence)
from pyomo.pysp.phutils import find_active_objective
from pyomo.pysp.scenariotree.manager_solver import \
    (ScenarioTreeManager,
     ScenarioTreeManagerClientSerial)
from pyomo.pysp.solutionioextensions import \
    (IPySPSolutionSaverExtension,
     IPySPSolutionLoaderExtension)
from pyomo.pysp.solutionwriter import ISolutionWriterExtension
from pyomo.pysp.ef import write_ef, create_ef_instance
from pyomo.pysp.solvers.spsolver import (SPSolver,
                                         SPSolverResults,
                                         SPSolverFactory)
from pyomo.pysp.scenariotree.manager import \
    ScenarioTreeManagerClientPyro
from pyomo.pysp.scenariotree.manager_solver import \
    ScenarioTreeManagerSolverClientPyro

logger = logging.getLogger('pyomo.pysp')

_ef_group_label = "EF Options"

class ExtensiveFormAlgorithm(PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()

        safe_declare_unique_option(
            options,
            "cvar_weight",
            PySPConfigValue(
                1.0,
                domain=_domain_nonnegative,
                description=(
                    "The weight associated with the CVaR term in "
                    "the risk-weighted objective "
                    "formulation. If the weight is 0, then "
                    "*only* a non-weighted CVaR cost will appear "
                    "in the EF objective - the expected cost "
                    "component will be dropped. Default is 1.0."
                ),
                doc=None,
                visibility=0),
            ap_group=_ef_group_label)
        safe_declare_unique_option(
            options,
            "generate_weighted_cvar",
            PySPConfigValue(
                False,
                domain=bool,
                description=(
                    "Add a weighted CVaR term to the "
                    "primary objective. Default is False."
                ),
                doc=None,
                visibility=0),
            ap_group=_ef_group_label)
        safe_declare_unique_option(
            options,
            "risk_alpha",
            PySPConfigValue(
                0.95,
                domain=_domain_unit_interval,
                description=(
                    "The probability threshold associated with "
                    "CVaR (or any future) risk-oriented "
                    "performance metrics. Default is 0.95."
                ),
                doc=None,
                visibility=0),
            ap_group=_ef_group_label)
        safe_declare_unique_option(
            options,
            "cc_alpha",
            PySPConfigValue(
                0.0,
                domain=_domain_unit_interval,
                description=(
                    "The probability threshold associated with a "
                    "chance constraint. The RHS will be one "
                    "minus this value. Default is 0."
                ),
                doc=None,
                visibility=0),
            ap_group=_ef_group_label)
        safe_declare_unique_option(
            options,
            "cc_indicator_var",
            PySPConfigValue(
                None,
                domain=_domain_must_be_str,
                description=(
                    "The name of the binary variable to be used "
                    "to construct a chance constraint. Default "
                    "is None, which indicates no chance "
                    "constraint."
                ),
                doc=None,
                visibility=0),
            ap_group=_ef_group_label)
        safe_declare_common_option(options,
                                   "solver")
        safe_declare_common_option(options,
                                   "solver_io")
        safe_declare_common_option(options,
                                   "solver_manager")
        safe_declare_common_option(options,
                                   "solver_options")
        safe_declare_common_option(options,
                                   "disable_warmstart")
        safe_declare_common_option(options,
                                   "solver_manager_pyro_host")
        safe_declare_common_option(options,
                                   "solver_manager_pyro_port")
        safe_declare_common_option(options,
                                   "solver_manager_pyro_shutdown")
        safe_declare_common_option(options,
                                   "verbose",
                                   ap_group=_ef_group_label)
        safe_declare_common_option(options,
                                   "output_times",
                                   ap_group=_ef_group_label)
        safe_declare_common_option(options,
                                   "output_solver_results",
                                   ap_group=_ef_group_label)

        return options

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.destroy_ef()
        if self._solver_manager is not None:
            if isinstance(self._solver_manager,
                          pyomo.solvers.plugins.smanager.\
                          pyro.SolverManager_Pyro):
                if self.get_option("pyro_shutdown_workers"):
                      self._solver_manager.shutdown_workers()
        self._solver_manager = None
        self._manager = None

    def __init__(self, manager, *args, **kwds):
        import pyomo.solvers.plugins.smanager.pyro
        super(ExtensiveFormAlgorithm, self).__init__(*args, **kwds)

        # TODO: after PH moves over to the new code
        #if not isinstance(manager, ScenarioTreeManager):
        #    raise TypeError("ExtensiveFormAlgorithm requires an instance of the "
        #                    "ScenarioTreeManager interface as the "
        #                    "second argument")
        if not manager.initialized:
            raise ValueError("ExtensiveFormAlgorithm requires a scenario tree "
                             "manager that has been fully initialized")

        self._manager = manager
        self.instance = None
        self._solver_manager = None
        self._solver = None

        # The following attributes will be modified by the
        # solve() method. For users that are scripting, these
        # can be accessed after the solve() method returns.
        # They will be reset each time solve() is called.
        ############################################
        self.objective = None
        self.gap = None
        self.termination_condition = None
        self.termination_message = None
        self.solver_status = None
        self.solution_status = None
        self.solver_results = None
        self.time = None
        self.pyomo_time = None
        ############################################

        # apparently the SolverFactory does not have sane
        # behavior when the solver name is None
        if self.get_option("solver") is None:
            raise ValueError("The 'solver' option can not be None")
        self._solver = SolverFactory(self.get_option("solver"),
                                     solver_io=self.get_option("solver_io"))
        if isinstance(self._solver, UnknownSolver):
            raise ValueError("Failed to create solver of type="+
                             self.get_option("solver")+
                             " for use in extensive form solve")

        solver_manager_type = self.get_option("solver_manager")
        if solver_manager_type == "phpyro":
            print("*** WARNING ***: PHPyro is not a supported solver "
                  "manager type for the extensive-form solver. "
                  "Falling back to serial.")
            solver_manager_type = 'serial'

        self._solver_manager = SolverManagerFactory(
            solver_manager_type,
            host=self.get_option("solver_manager_pyro_host"),
            port=self.get_option("solver_manager_pyro_port"))
        if self._solver_manager is None:
            raise ValueError("Failed to create solver manager of type="
                             +self.get_option("solver")+
                             " for use in extensive form solve")

    def build_ef(self):
        self.destroy_ef()
        if self.get_option("verbose"):
            print("Creating extensive form instance")
        start_time = time.time()

        # then validate the associated parameters.
        generate_weighted_cvar = False
        cvar_weight = None
        risk_alpha = None
        if self.get_option("generate_weighted_cvar"):
            generate_weighted_cvar = True
            cvar_weight = self.get_option("cvar_weight")
            risk_alpha = self.get_option("risk_alpha")

        self.instance = create_ef_instance(
            self._manager.scenario_tree,
            verbose_output=self.get_option("verbose"),
            generate_weighted_cvar=generate_weighted_cvar,
            cvar_weight=cvar_weight,
            risk_alpha=risk_alpha,
            cc_indicator_var_name=self.get_option("cc_indicator_var"),
            cc_alpha=self.get_option("cc_alpha"))

        if self.get_option("verbose") or self.get_option("output_times"):
            print("Time to construct extensive form instance=%.2f seconds"
                  %(time.time() - start_time))

    def destroy_ef(self):
        if self.instance is not None:
            for scenario in self._manager.scenario_tree.scenarios:
                self.instance.del_component(scenario.name)
                scenario._instance_objective.activate()
        self.instance = None

    def write(self, filename):

        if self.instance is None:
            raise RuntimeError(
                "The extensive form instance has not been constructed."
                "Call the build_ef() method to construct it.")

        suf = os.path.splitext(filename)[1]
        if suf not in ['.nl','.lp','.mps']:
            if self._solver.problem_format() == ProblemFormat.cpxlp:
                filename += '.lp'
            elif self._solver.problem_format() == ProblemFormat.nl:
                filename += '.nl'
            elif self._solver.problem_format() == ProblemFormat.mps:
                filename += '.mps'
            else:
                raise ValueError("Could not determine output file format. "
                                 "No recognized ending suffix was provided "
                                 "and no format was indicated was by the "
                                 "--solver-io option.")

        start_time = time.time()
        print("Writing extensive form to file="+filename)
        smap_id = write_ef(self.instance,
                           filename,
                           self.get_option("symbolic_solver_labels"))

        if self.get_option("verbose") or self.get_option("output_times"):
            print("Time to write output file=%.2f seconds"
                  % (time.time() - start_time))

        return filename, smap_id

    def solve(self,
              check_status=True,
              exception_on_failure=True,
              output_solver_log=False,
              symbolic_solver_labels=False,
              keep_solver_files=False,
              io_options=None):
        # TODO: Does this import need to be delayed because
        #       it is in a plugins subdirectory
        from pyomo.solvers.plugins.solvers.persistent_solver import \
            PersistentSolver

        if self.instance is None:
            raise RuntimeError(
                "The extensive form instance has not been constructed."
                "Call the build_ef() method to construct it.")

        start_time = time.time()
        if self.get_option("verbose"):
            print("Queuing extensive form solve")

        self.objective = None
        self.gap = None
        self.bound = None
        self.termination_condition = None
        self.termination_message = None
        self.solver_status = None
        self.solution_status = None
        self.solver_results = None
        self.time = None
        self.pyomo_time = None

        if isinstance(self._solver, PersistentSolver):
            self._solver.compile_instance(
                self.instance,
                symbolic_solver_labels=symbolic_solver_labels)

        solve_kwds = {}
        solve_kwds['load_solutions'] = False
        if keep_solver_files:
            solve_kwds['keepfiles'] = True
        if symbolic_solver_labels:
            solve_kwds['symbolic_solver_labels'] = True
        if output_solver_log:
            solve_kwds['tee'] = True

        solver_options = self.get_option("solver_options")
        if len(solver_options) > 0:
            if type(solver_options) is tuple:
                solve_kwds["options"] = {}
                for name_val in solver_options:
                    assert "=" in name_val
                    name, val = name_val.split("=")
                    solve_kwds["options"][name.strip()] = val.strip()
            else:
                solve_kwds["options"] = solver_options

        if io_options is not None:
            solve_kwds.update(io_options)

        self.objective_sense = \
            find_active_objective(self.instance).sense

        if (not self.get_option("disable_warmstart")) and \
           (self._solver.warm_start_capable()):
            action_handle = self._solver_manager.queue(self.instance,
                                                       opt=self._solver,
                                                       warmstart=True,
                                                       **solve_kwds)
        else:
            action_handle = self._solver_manager.queue(self.instance,
                                                       opt=self._solver,
                                                       **solve_kwds)

        if self.get_option("verbose"):
            print("Waiting for extensive form solve")
        results = self._solver_manager.wait_for(action_handle)

        if self.get_option("verbose"):
            print("Done with extensive form solve - loading results")

        if self.get_option("output_solver_results"):
            print("Results for ef:")
            results.write(num=1)

        self.solver_results = results
        if hasattr(results.solver,"user_time") and \
           (not isinstance(results.solver.user_time,
                           UndefinedData)) and \
           (results.solver.user_time is not None):
            # the solve time might be a string, or might
            # not be - we eventually would like more
            # consistency on this front from the solver
            # plugins.
            self.time = \
                float(results.solver.user_time)
        elif hasattr(results.solver,"time"):
            self.time = \
                float(results.solver.time)
        else:
            self.time = None

        if hasattr(results,"pyomo_solve_time"):
            self.pyomo_time = \
                results.pyomo_solve_time
        else:
            self.pyomo_time = None

        self.termination_condition = \
            results.solver.termination_condition
        self.termination_message = None
        if hasattr(results.solver,"termination_message"):
            self.termination_message = results.solver.termination_message
        elif hasattr(results.solver,"message"):
            self.termination_message = results.solver.message
        self.solver_status = \
            results.solver.status

        if len(results.solution) > 0:
            assert len(results.solution) == 1

            results_sm = results._smap
            self.instance.solutions.load_from(results)

            solution0 = results.solution(0)
            if hasattr(solution0, "gap") and \
               (solution0.gap is not None) and \
               (not isinstance(solution0.gap,
                               UndefinedData)):
                self.gap = float(solution0.gap)
            else:
                self.gap = None

            self.solution_status = solution0.status

            if self.get_option("verbose"):
                print("Storing solution in scenario tree")

            for scenario in self._manager.scenario_tree.scenarios:
                scenario.update_solution_from_instance()
            self._manager.scenario_tree.snapshotSolutionFromScenarios()
            self.objective = self._manager.scenario_tree.\
                             findRootNode().\
                             computeExpectedNodeCost()
            if self.gap is not None:
                if self.objective_sense == pyomo.core.base.minimize:
                    self.bound = self.objective - self.gap
                else:
                    self.bound = self.objective + self.gap

        else:

            self.objective = None
            self.gap = None
            self.bound = None
            self.solution_status = None

        failure = False

        if check_status:
            if not ((self.solution_status == SolutionStatus.optimal) or \
                    (self.solution_status == SolutionStatus.feasible)):
                failure = True
                if self.get_option("verbose") or \
                   exception_on_failure:
                    msg = ("EF solve failed solution status check:\n"
                           "Solver Status: %s\n"
                           "Termination Condition: %s\n"
                           "Solution Status: %s\n"
                           % (self.solver_status,
                              self.termination_condition,
                              self.solution_status))
                    if self.get_option("verbose"):
                        print(msg)
                    if exception_on_failure:
                        raise RuntimeError(msg)
        else:
            if self.get_option("verbose"):
                print("EF solve completed. Skipping status check.")

        if self.get_option("verbose") or self.get_option("output_times"):
            print("Time to solve and load results for the "
                  "extensive form=%.2f seconds"
                  % (time.time()-start_time))

        return failure

class EFSolver(SPSolver, PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        return ExtensiveFormAlgorithm.register_options(options)

    def __init__(self):
        super(EFSolver, self).__init__(self.register_options())
        self.set_options_to_default()

    def set_options_to_default(self):
        self._options = self.register_options()

    @property
    def options(self):
        return self._options

    @property
    def name(self):
        return "ef"

    def _solve_impl(self,
                    sp,
                    output_solver_log=False,
                    keep_solver_files=False,
                    symbolic_solver_labels=False):
        """
        Solve a stochastic program by building the extensive
        form and calling and a Pyomo solver.

        See the 'solve' method on the base class for
        additional keyword documentation.

        Args:
            sp: The stochastic program to solve. Pyro based
                managers are not accepted. All scenario
                models must be managed locally.
            output_solver_log (bool): Stream the solver
                output during the solve.
            keep_solver_files (bool): Retain temporary solver
                input and output files after the solve completes.
            symbolic_solver_labels (bool): Generate solver
                input files using human-readable symbols
                (makes debugging easier).

        Returns: A results object with information about the solution.
        """

        if isinstance(sp,
                      (ScenarioTreeManagerClientPyro,
                       ScenarioTreeManagerSolverClientPyro)):
            raise TypeError("The EF solver does not handle "
                            "Pyro-based scenario tree managers")

        orig_parents = {}
        if sp.scenario_tree.contains_bundles():
            for scenario in sp.scenario_tree.scenarios:
                if scenario._instance._parent is not None:
                    orig_parents[scenario] = scenario._instance._parent
                    scenario._instance._parent = None
                    assert not scenario._instance_objective.active
                    scenario._instance_objective.activate()
        try:
            with ExtensiveFormAlgorithm(sp, self._options) as ef:
                ef.build_ef()
                ef.solve(
                    output_solver_log=output_solver_log,
                    keep_solver_files=keep_solver_files,
                    symbolic_solver_labels=symbolic_solver_labels,
                    check_status=False)
        finally:
            for scenario, parent in orig_parents.items():
                scenario._instance._parent = parent
                assert scenario._instance_objective.active
                scenario._instance_objective.deactivate()

        results = SPSolverResults()
        results.objective = ef.objective
        results.bound = ef.bound
        results.status = ef.solution_status
        results.solver.status = ef.solver_status
        results.solver.termination_condition = ef.termination_condition
        results.solver.message = ef.termination_message
        results.solver.time = ef.time
        results.solver.pyomo_time = ef.pyomo_time
        results.xhat = None
        if ef.solution_status is not None:
            xhat = results.xhat = {}
            for stage in sp.scenario_tree.stages[:-1]:
                for node in stage.nodes:
                    node_xhat = xhat[node.name] = {}
                    reference_scenario = node.scenarios[0]
                    node_x = reference_scenario._x[node.name]
                    for id_ in node._variable_ids:
                        node_xhat[id_] = node_x[id_]
                    # TODO: stage costs
                    #for stagenum, stage in enumerate(sp.scenario_tree.stages[:-1]):
                    #    cost_variable_name, cost_variable_index = \
                    #        stage._cost_variable
                    #    stage_cost_obj = \
                    #        instance.find_component(cost_variable_name)[cost_variable_index]
                    #    if not stage_cost_obj.is_expression_type():
                    #        refvar = ComponentUID(stage_cost_obj,cuid_buffer=tmp).\
                    #            find_component(reference_model)
                    #        refvar.value = stage_cost_obj.value
                    #        refvar.stale = stage_cost_obj.stale

        return results

def runef_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    EFSolver.register_options(options)
    ScenarioTreeManagerClientSerial.register_options(options)
    safe_register_common_option(options,
                                "verbose")
    safe_register_common_option(options,
                                "disable_gc")
    safe_register_common_option(options,
                                "profile")
    safe_register_common_option(options,
                                "traceback")
    safe_register_common_option(options,
                                "symbolic_solver_labels")
    safe_register_common_option(options,
                                "output_solver_log")
    safe_register_common_option(options,
                                "keep_solver_files")
    safe_register_common_option(options,
                                "output_scenario_tree_solution")
    safe_register_common_option(options,
                                "solution_saver_extension")
    safe_register_common_option(options,
                                "solution_loader_extension")
    safe_register_unique_option(
        options,
        "solution_writer",
        PySPConfigValue(
            (),
            domain=_domain_tuple_of_str,
            description=(
                "The name of a python module specifying a user-defined "
                "plugin implementing the ISolutionWriterExtension "
                "interface. Invoked to save a scenario tree solution. Use "
                "this option when generating a template configuration file "
                "or invoking command-line help in order to include any "
                "plugin-specific options. This option can used multiple "
                "times from the command line to specify more than one "
                "plugin."
            ),
            doc=None,
            visibility=0),
        ap_kwds={'action': 'append'},
        ap_group=_extension_options_group_title)
    safe_register_unique_option(
        options,
        "output_file",
        PySPConfigValue(
            None,
            domain=_domain_must_be_str,
            description=(
                "The name of the extensive form output file "
                "(currently LP, MPS, and NL file formats are "
                "supported). If the option value does not end "
                "in '.lp', '.mps', or '.nl', then the output format "
                "will be inferred from the settings for the chosen "
                "solver interface, and the appropriate suffix "
                "will be appended to the name. Use of this option "
                "will disable the solve."
            ),
            doc=None,
            visibility=0),
        ap_group=_output_options_group_title)
    safe_register_unique_option(
        options,
        "output_scenario_costs",
        PySPConfigValue(
            None,
            domain=_domain_must_be_str,
            description=(
                "A file name where individual scenario costs from the "
                "solution will be stored. The format is determined "
                "from the extension used in the filename. Recognized "
                "extensions: [.csv, .json, .yaml]"
            ),
            doc=None,
            visibility=0))

    #
    # Deprecated
    #

    class _DeprecatedActivateJSONIOSolutionSaver(
            pyutilib.misc.config.argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            if nargs is not None:
                raise ValueError("nargs not allowed")
            super(_DeprecatedActivateJSONIOSolutionSaver, self).\
                __init__(option_strings, dest, nargs=0, **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            logger.warning(
                "DEPRECATED: The '--activate-json-io-solution-saver "
                "command-line option has been deprecated and will "
                "be removed in the future. Please the following instead: "
                "'----solution-saver-extension=pyomo.pysp.plugins.jsonio'")
            val = getattr(namespace,
                          'CONFIGBLOCK.solution_saver_extension', [])
            setattr(namespace,
                    'CONFIGBLOCK.solution_saver_extension',
                    val + ["pyomo.pysp.plugins.jsonio"])

    def _warn_activate_jsonio_solution_saver(val):
        # don't use logger here since users might not import
        # the pyomo logger in a scripting interface
        sys.stderr.write(
            "\tWARNING: The 'activate_jsonio_solution_saver' "
            "config item will be ignored "
            "unless it is being used as a command-line option "
            "where it can be redirected to 'solution_saver_extension'. "
            "Please use 'solution_saver_extension=pyomo.pysp.plugins.jsonio' "
            "instead.\n")
        return _domain_tuple_of_str(val)

    safe_declare_unique_option(
        options,
        "activate_jsonio_solution_saver",
        PySPConfigValue(
            None,
            domain=_warn_activate_jsonio_solution_saver,
            description=(
                "Deprecated alias for "
                "--solution-saver-extension=pyomo.pysp.plugins.jsonio"
            ),
            doc=None,
            visibility=1),
        ap_kwds={'action':_DeprecatedActivateJSONIOSolutionSaver},
        ap_group=_deprecated_options_group_title,
        declare_for_argparse=True)

    return options

#
# Construct a scenario tree manager and an
# ExtensiveFormAlgorithm to solve it.
#

def runef(options,
          solution_loaders=(),
          solution_savers=(),
          solution_writers=()):
    """
    Construct a senario tree manager and solve it
    with the Extensive Form solver.
    """
    start_time = time.time()

    solution_loaders = sort_extensions_by_precedence(solution_loaders)
    solution_savers = sort_extensions_by_precedence(solution_savers)
    solution_writers = sort_extensions_by_precedence(solution_writers)

    with ScenarioTreeManagerClientSerial(options) as sp:
        sp.initialize()

        for plugin in solution_loaders:
            ret = plugin.load(sp)
            if not ret:
                logger.warning(
                    "Loader extension %s call did not return "
                    "True. This might indicate failure to load data."
                    % (plugin))

        if options.output_file is not None:
            with ExtensiveFormAlgorithm(sp, options) as ef:
                ef.build_ef()
                ef.write(filename)
        else:
            print("")
            print("Running the EF solver for "
                  "stochastic programming problems.")
            ef = EFSolver()
            ef_options = ef.extract_user_options_to_dict(options,
                                                         sparse=True)
            results = ef.solve(
                sp,
                options=ef_options,
                output_solver_log=options.output_solver_log,
                keep_solver_files=options.keep_solver_files,
                symbolic_solver_labels=options.symbolic_solver_labels)
            xhat = results.xhat
            del results.xhat
            print("")
            print(results)
            results.xhat = xhat

            if options.output_scenario_tree_solution:
                print("")
                sp.scenario_tree.snapshotSolutionFromScenarios()
                sp.scenario_tree.pprintSolution()
                sp.scenario_tree.pprintCosts()

            if options.output_scenario_costs is not None:
                if options.output_scenario_costs.endswith('.json'):
                    import json
                    result = {}
                    for scenario in sp.scenario_tree.scenarios:
                        result[str(scenario.name)] = scenario._cost
                    with open(options.output_scenario_costs, 'w') as f:
                        json.dump(result, f, indent=2, sort_keys=True)
                elif options.output_scenario_costs.endswith('.yaml'):
                    import yaml
                    result = {}
                    for scenario in sp.scenario_tree.scenarios:
                        result[str(scenario.name)] = scenario._cost
                    with open(options.output_scenario_costs, 'w') as f:
                        yaml.dump(result, f)
                else:
                    if not options.output_scenario_costs.endswith('.csv'):
                        print("Unrecognized file extension. Using CSV "
                              "format to store scenario costs")
                    with open(options.output_scenario_costs, 'w') as f:
                        for scenario in sp.scenario_tree.scenarios:
                            f.write("%s,%r\n"
                                    % (scenario.name,scenario._cost))

            for plugin in solution_savers:
                if not plugin.save(sp):
                    logger.warning("Saver extension %s call did not "
                          "return True. This might indicate failure "
                          "to save data." % (plugin))

            for plugin in solution_writers:
                plugin.write(sp.scenario_tree, "ef")

    print("")
    print("Total EF execution time=%.2f seconds"
          % (time.time() - start_time))
    print("")

    return 0

#
# The main driver routine for the runef script
#

def main(args=None):
    #
    # Top-level command that executes everything
    #

    #
    # Import plugins
    #
    import pyomo.environ

    #
    # Parse command-line options.
    #
    try:
        options, extensions = parse_command_line(
            args,
            runef_register_options,
            with_extensions={'solution_loader_extension':
                             IPySPSolutionLoaderExtension,
                             'solution_saver_extension':
                             IPySPSolutionSaverExtension,
                             'solution_writer':
                             ISolutionWriterExtension},
            prog='runef',
            description=(
"""Optimize a stochastic program using the Extensive Form (EF) solver."""
            ))

    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(runef,
                          options,
                          cmd_kwds={'solution_loaders':
                                    extensions['solution_loader_extension'],
                                    'solution_savers':
                                    extensions['solution_saver_extension'],
                                    'solution_writers':
                                    extensions['solution_writer']},
                          error_label="runef: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

SPSolverFactory.register_solver("ef", EFSolver)

if __name__ == "__main__":
    sys.exit(main())
