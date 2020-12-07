#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import logging
import sys
import time
import argparse

import pyomo.solvers
from pyomo.common.dependencies import yaml
from pyomo.common import pyomo_command
from pyomo.opt import (SolverFactory,
                       undefined,
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
                                    _domain_nonnegative,
                                    _domain_must_be_str,
                                    _domain_unit_interval,
                                    _domain_tuple_of_str,
                                    _output_options_group_title,
                                    _extension_options_group_title,
                                    _deprecated_options_group_title)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command,
                                  sort_extensions_by_precedence)
from pyomo.pysp.phutils import find_active_objective
from pyomo.pysp.scenariotree.manager_solver import \
    (ScenarioTreeManagerClientSerial)
from pyomo.pysp.solutionioextensions import \
    (IPySPSolutionSaverExtension,
     IPySPSolutionLoaderExtension)
from pyomo.pysp.solutionwriter import ISolutionWriterExtension
from pyomo.pysp.ef import write_ef, create_ef_instance

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
        safe_declare_unique_option(
            options,
            "mipgap",
            PySPConfigValue(
                None,
                domain=_domain_unit_interval,
                description=(
                    "Specifies the mipgap for the EF solve."
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
                                   "pyro_host")
        safe_declare_common_option(options,
                                   "pyro_port")
        safe_declare_common_option(options,
                                   "pyro_shutdown")
        safe_declare_common_option(options,
                                   "verbose",
                                   ap_group=_ef_group_label)
        safe_declare_common_option(options,
                                   "output_times",
                                   ap_group=_ef_group_label)
        safe_declare_common_option(options,
                                   "output_solver_results",
                                   ap_group=_ef_group_label)
        safe_declare_common_option(options,
                                   "symbolic_solver_labels",
                                   ap_group=_ef_group_label)
        safe_declare_common_option(options,
                                   "output_solver_log",
                                   ap_group=_ef_group_label)
        safe_declare_common_option(options,
                                   "verbose",
                                   ap_group=_ef_group_label)
        safe_declare_common_option(options,
                                   "output_times",
                                   ap_group=_ef_group_label)
        safe_declare_common_option(options,
                                   "keep_solver_files",
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
        self.objective = undefined
        self.objective_sense = undefined
        self.gap = undefined
        self.termination_condition = undefined
        self.solver_status = undefined
        self.solution_status = undefined
        self.solver_results = undefined
        self.pyomo_solve_time = undefined
        self.solve_time = undefined

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
        self.objective = undefined
        self.gap = undefined
        self.termination_condition = undefined
        self.solver_status = undefined
        self.solution_status = undefined
        self.solver_results = undefined
        self.pyomo_solve_time = undefined
        self.solve_time = undefined
        ############################################

        self._solver = SolverFactory(self.get_option("solver"),
                                     solver_io=self.get_option("solver_io"))
        if isinstance(self._solver, UnknownSolver):
            raise ValueError("Failed to create solver of type="+
                             self.get_option("solver")+
                             " for use in extensive form solve")
        if len(self.get_option("solver_options")) > 0:
            if self.get_option("verbose"):
                print("Initializing ef solver with options="
                      +str(list(self.get_option("solver_options"))))
            self._solver.set_options("".join(self.get_option("solver_options")))
        if self.get_option("mipgap") is not None:
            if (self.get_option("mipgap") < 0.0) or \
               (self.get_option("mipgap") > 1.0):
                raise ValueError("Value of the mipgap parameter for the EF "
                                 "solve must be on the unit interval; "
                                 "value specified="+str(self.get_option("mipgap")))
            self._solver.options.mipgap = float(self.get_option("mipgap"))

        solver_manager_type = self.get_option("solver_manager")
        if solver_manager_type == "phpyro":
            print("*** WARNING ***: PHPyro is not a supported solver "
                  "manager type for the extensive-form solver. "
                  "Falling back to serial.")
            solver_manager_type = 'serial'

        self._solver_manager = SolverManagerFactory(
            solver_manager_type,
            host=self.get_option("pyro_host"),
            port=self.get_option("pyro_port"))
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
        if self.get_option("verbose"):
            print("Starting to write extensive form")

        smap_id = write_ef(self.instance,
                           filename,
                           self.get_option("symbolic_solver_labels"))

        print("Extensive form written to file="+filename)
        if self.get_option("verbose") or self.get_option("output_times"):
            print("Time to write output file=%.2f seconds"
                  % (time.time() - start_time))

        return filename, smap_id

    def solve(self,
              check_status=True,
              exception_on_failure=True,
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

        self.objective = undefined
        self.gap = undefined
        self.bound = undefined
        self.pyomo_solve_time = undefined
        self.solve_time = undefined
        self.termination_condition = undefined
        self.solver_status = undefined
        self.solution_status = undefined
        self.solver_results = undefined

        if isinstance(self._solver, PersistentSolver):
            self._solver.set_instance(self.instance,
                                      symbolic_solver_labels=self.get_option("symbolic_solver_labels"))

        solve_kwds = {}
        solve_kwds['load_solutions'] = False
        if self.get_option("keep_solver_files"):
            solve_kwds['keepfiles'] = True
        if self.get_option("symbolic_solver_labels"):
            solve_kwds['symbolic_solver_labels'] = True
        if self.get_option("output_solver_log"):
            solve_kwds['tee'] = True

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
            self.solve_time = \
                float(results.solver.user_time)
        elif hasattr(results.solver,"time"):
            self.solve_time = \
                float(results.solver.time)
        else:
            self.solve_time = undefined

        if hasattr(results,"pyomo_solve_time"):
            self.pyomo_solve_time = \
                results.pyomo_solve_time
        else:
            self.pyomo_solve_times = undefined

        self.termination_condition = \
            results.solver.termination_condition
        self.solver_status = \
            results.solver.status

        if len(results.solution) > 0:
            assert len(results.solution) == 1

            results_sm = results._smap
            self.instance.solutions.load_from(results)

            solution0 = results.solution(0)
            if hasattr(solution0, "gap") and \
               (solution0.gap is not None):
                self.gap = solution0.gap
            else:
                self.gap = undefined

            self.solution_status = solution0.status

            if self.get_option("verbose"):
                print("Storing solution in scenario tree")

            for scenario in self._manager.scenario_tree.scenarios:
                scenario.update_solution_from_instance()
            self._manager.scenario_tree.snapshotSolutionFromScenarios()
            self.objective = self._manager.scenario_tree.\
                             findRootNode().\
                             computeExpectedNodeCost()
            if self.gap is not undefined:
                if self.objective_sense == pyomo.core.base.minimize:
                    self.bound = self.objective - self.gap
                else:
                    self.bound = self.objective + self.gap

        else:

            self.objective = undefined
            self.gap = undefined
            self.bound = undefined
            self.solution_status = undefined

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

def runef_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()

    safe_register_common_option(options,
                               "verbose")
    safe_register_common_option(options,
                               "disable_gc")
    safe_register_common_option(options,
                               "profile")
    safe_register_common_option(options,
                               "traceback")
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
                "times from the command line to specify more than one plugin."
            ),
            doc=None,
            visibility=0),
        ap_kwds={'action': 'append'},
        ap_group=_extension_options_group_title)
    safe_register_unique_option(
        options,
        "output_file",
        PySPConfigValue(
            "efout",
            domain=_domain_must_be_str,
            description=(
                "The name of the extensive form output file "
                "(currently LP, MPS, and NL file formats are "
                "supported). If the option value does not end "
                "in '.lp', '.mps', or '.nl', then the output format "
                "will be inferred from the settings for the --solver "
                "and --solver-io options, and the appropriate suffix "
                "will be appended to the name. Default is 'efout'."
            ),
            doc=None,
            visibility=0),
        ap_group=_output_options_group_title)
    safe_register_unique_option(
        options,
        "solve",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Solve the extensive form model. Default is "
                "False, which implies that the EF will be "
                "saved to a file."
            ),
            doc=None,
            visibility=0))
    safe_register_unique_option(
        options,
        "output_scenario_costs",
        PySPConfigValue(
            None,
            domain=_domain_must_be_str,
            description=(
                "A file name where individual scenario costs from the solution "
                "will be stored. The format is determined from the extension used "
                "in the filename. Recognized extensions: [.csv, .json, .yaml]"
            ),
            doc=None,
            visibility=0))
    ScenarioTreeManagerClientSerial.register_options(options)
    ExtensiveFormAlgorithm.register_options(options)

    #
    # Deprecated
    #

    # this will cause the deprecated "shutdown_pyro" version
    # to appear
    safe_register_common_option(options,
                                "pyro_shutdown")
    # this will cause the deprecated "shutdown_pyro_workers"
    # version to appear
    safe_register_common_option(options,
                                "pyro_shutdown_workers")

    class _DeprecatedActivateJSONIOSolutionSaver(argparse.Action):
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

    import pyomo.environ

    start_time = time.time()

    solution_loaders = sort_extensions_by_precedence(solution_loaders)
    solution_savers = sort_extensions_by_precedence(solution_savers)
    solution_writers = sort_extensions_by_precedence(solution_writers)

    with ScenarioTreeManagerClientSerial(options) \
         as manager:
        manager.initialize()

        loaded = False
        for plugin in solution_loaders:
            ret = plugin.load(manager)
            if not ret:
                print("WARNING: Loader extension %s call did not return True. "
                      "This might indicate failure to load data." % (plugin))
            else:
                loaded = True

        print("")
        print("Initializing extensive form algorithm for "
              "stochastic programming problems.")
        with ExtensiveFormAlgorithm(manager, options) as ef:

            ef.build_ef()
            # This is somewhat of a hack to get around the
            # weird semantics of this script (assumed by tests)
            if (not options.solve) or \
               (options.get("output_file")._userSet):

                ef.write(options.output_file)

            if not options.solve:

                if options.output_scenario_costs is not None:
                    print("WARNING: output_scenario_costs option "
                          "will be ignored because the extensive form "
                          "has not been solved.")
                if len(solution_savers):
                    print("WARNING: Solution saver extensions will "
                          "not be called because the extensive form "
                          "has not been solved.")
                if len(solution_writers):
                    print("WARNING: Solution writer extensions will "
                          "not be called because the extensive form "
                          "has not been solved.")

            else:
                ef.solve()

                print("EF solve completed and solution status is %s"
                      % ef.solution_status)
                print("EF solve termination condition is %s"
                      % ef.termination_condition)
                print("EF objective: %12.5f" % ef.objective)
                if ef.gap is not undefined:
                    print("EF gap:       %12.5f" % ef.gap)
                    print("EF bound:     %12.5f" % ef.bound)
                else:
                    assert ef.bound is undefined
                    print("EF gap:       <unknown>")
                    print("EF bound:     <unknown>")

                # handle output of solution from the scenario tree.
                print("")
                print("Extensive form solution:")
                manager.scenario_tree.pprintSolution()
                print("")
                print("Extensive form costs:")
                manager.scenario_tree.pprintCosts()

                if options.output_scenario_tree_solution:
                    print("Final solution (scenario tree format):")
                    manager.scenario_tree.pprintSolution()

                if options.output_scenario_costs is not None:
                    if options.output_scenario_costs.endswith('.json'):
                        import json
                        result = {}
                        for scenario in manager.scenario_tree.scenarios:
                            result[str(scenario.name)] = scenario._cost
                        with open(options.output_scenario_costs, 'w') as f:
                            json.dump(result, f, indent=2, sort_keys=True)
                    elif options.output_scenario_costs.endswith('.yaml'):
                        result = {}
                        for scenario in manager.scenario_tree.scenarios:
                            result[str(scenario.name)] = scenario._cost
                        with open(options.output_scenario_costs, 'w') as f:
                            yaml.dump(result, f)
                    else:
                        if not options.output_scenario_costs.endswith('.csv'):
                            print("Unrecognized file extension. Using CSV format "
                                  "to store scenario costs")
                        with open(options.output_scenario_costs, 'w') as f:
                            for scenario in manager.scenario_tree.scenarios:
                                f.write("%s,%r\n" % (scenario.name,scenario._cost))


                for plugin in solution_savers:
                    if not plugin.save(manager):
                        print("WARNING: Saver extension %s call did not "
                              "return True. This might indicate failure "
                              "to save data." % (plugin))

                for plugin in solution_writers:
                    plugin.write(manager.scenario_tree, "ef")

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
"""Construct and Solve an Extensive Form for Stochastic Programs"""
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

@pyomo_command('runef',
               'Convert a stochastic program to extensive form and optimize')
def EF_main(args=None):
    return main(args=args)
