#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ScenarioTreePreprocessor",)

# TODO: Verify what needs to be done for persistent solver plugins
#       when advanced preprocessing is disabled and finish
#       implementing for that option.
import time

# these are the only two preprocessors currently invoked by the
# simple_preprocessor, which in turn is invoked by the preprocess()
# method of PyomoModel.
from pyomo.opt import ProblemFormat, PersistentSolver
from pyomo.repn.canonical_repn import LinearCanonicalRepn
from pyomo.repn.compute_canonical_repn import preprocess_block_objectives \
    as canonical_preprocess_block_objectives
from pyomo.repn.compute_canonical_repn import preprocess_block_constraints \
    as canonical_preprocess_block_constraints
from pyomo.repn.compute_canonical_repn import preprocess_constraint \
    as canonical_preprocess_constraint
from pyomo.repn.compute_ampl_repn import preprocess_block_objectives \
    as ampl_preprocess_block_objectives
from pyomo.repn.compute_ampl_repn import preprocess_block_constraints \
    as ampl_preprocess_block_constraints
from pyomo.repn.compute_ampl_repn import preprocess_constraint \
    as ampl_preprocess_constraint
from pyomo.repn.ampl_repn import generate_ampl_repn
from pyomo.repn.canonical_repn import generate_canonical_repn
import pyomo.util
from pyomo.pysp.util.config import (PySPConfigBlock,
                                    safe_declare_common_option)
from pyomo.pysp.util.configured_object import PySPConfiguredObject

from six import iteritems, itervalues
from six.moves import xrange

canonical_expression_preprocessor = \
    pyomo.util.PyomoAPIFactory("pyomo.repn.compute_canonical_repn")
ampl_expression_preprocessor = \
    pyomo.util.PyomoAPIFactory("pyomo.repn.compute_ampl_repn")


#
# We only want to do the minimal amount of work to get the instance
# back to a consistent "preprocessed" state. The following attributes
# are introduced to help perform the minimal amount of work, and
# should be augmented in the future if we can somehow do less. These
# attributes are initially cleared, and are re-set - following
# preprocessing, if necessary - before each round of model I/O.
#
class ScenarioTreePreprocessor(PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "ScenarioTreePreprocessor class")

    safe_declare_common_option(_declared_options,
                               "disable_advanced_preprocessing")
    safe_declare_common_option(_declared_options,
                               "preprocess_fixed_variables")
    safe_declare_common_option(_declared_options,
                               "symbolic_solver_labels")

    #
    # various
    #
    safe_declare_common_option(_declared_options,
                               "output_times")
    safe_declare_common_option(_declared_options,
                               "verbose")

    def __init__(self, *args, **kwds):

        super(ScenarioTreePreprocessor, self).__init__(*args, **kwds)
        self._scenario_solver = {}
        self._scenario_instance = {}
        self._scenario_objective = {}

        #
        # Bundle related objects
        #
        self._bundle_instances = {}
        self._bundle_solvers = {}
        self._bundle_scenarios = {}
        self._scenario_to_bundle_map = {}
        self._bundle_first_preprocess = {}

        # maps between instance name and a list of variable (name,
        # index) pairs
        self.fixed_variables = {}
        self.freed_variables = {}
        # indicates update status of instances since the last
        # preprocessing round
        self.objective_updated = {}
        self.all_constraints_updated = {}
        self.constraints_updated_list = {}

    def add_scenario(self, scenario, scenario_instance, scenario_solver):

        assert scenario._name not in self._scenario_instance
        assert scenario._name not in self._scenario_to_bundle_map

        self._scenario_instance[scenario._name] = scenario_instance
        self._scenario_solver[scenario._name] = scenario_solver
        self._scenario_objective[scenario._name] = scenario._instance_objective

        self.fixed_variables[scenario._name] = []
        self.freed_variables[scenario._name] = []
        self.objective_updated[scenario._name] = True
        self.all_constraints_updated[scenario._name] = True
        self.constraints_updated_list[scenario._name] = []

        self.objective_updated[scenario._name] = True
        self.all_constraints_updated[scenario._name] = True

        if not self._options.disable_advanced_preprocessing:
            scenario_instance = self._scenario_instance[scenario._name]
            for block in scenario_instance.block_data_objects(active=True):
                block._gen_obj_ampl_repn = False
                block._gen_con_ampl_repn = False
                block._gen_obj_canonical_repn = False
                block._gen_con_canonical_repn = False

    def remove_scenario(self, scenario):

        assert scenario._name in self._scenario_instance
        assert scenario._name not in self._scenario_to_bundle_map

        if self._options.disable_advanced_preprocessing:
            scenario_instance = self._scenario_instance[scenario_name]
            for block in scenario_instance.block_data_objects(active=True):
                block._gen_obj_ampl_repn = False
                block._gen_con_ampl_repn = False
                block._gen_obj_canonical_repn = False
                block._gen_con_canonical_repn = False

        del self._scenario_instance[scenario._name]
        del self._scenario_solver[scenario._name]

        del self.fixed_variables[scenario._name]
        del self.freed_variables[scenario._name]
        del self.objective_updated[scenario._name]
        del self.all_constraints_updated[scenario._name]
        del self.constraints_updated_list[scenario._name]

        del self.objective_updated[scenario._name]
        del self.all_constraints_updated[scenario._name]

    def add_bundle(self, bundle, bundle_instance, bundle_solver):

        assert bundle._name not in self._bundle_instances

        self._bundle_instances[bundle._name] = bundle_instance
        self._bundle_solvers[bundle._name] = bundle_solver
        self._bundle_scenarios[bundle._name] = list(bundle._scenario_names)
        self._bundle_first_preprocess[bundle._name] = True

        for scenario_name in self._bundle_scenarios[bundle._name]:
            assert scenario_name in self._scenario_instance
            assert scenario_name not in self._scenario_to_bundle_map
            self._scenario_to_bundle_map[scenario_name] = bundle._name

    def remove_bundle(self, bundle):

        assert bundle._name in self._bundle_instances

        for scenario_name in self._bundle_scenarios[bundle._name]:
            assert scenario_name in self._scenario_instance
            assert scenario_name in self._scenario_to_bundle_map
            self._scenario_to_bundle_map[scenario_name] = bundle._name

        del self._bundle_instances[bundle._name]
        del self._bundle_solvers[bundle._name]
        del self._bundle_scenarios[bundle._name]
        del self._bundle_first_preprocess[bundle._name]

    def clear_update_flags(self, name=None):
        if name is not None:
            self.objective_updated[name] = False
            self.all_constraints_updated[name] = False
            self.constraints_updated_list[name] = []
        else:
            for key in self.instances:
                self.objective_updated[key] = False
                self.all_constraints_updated[key] = False
                self.constraints_updated_list[key] = []

    def has_fixed_variables(self, name=None):
        if name is None:
            for val in itervalues(self.fixed_variables):
                if len(val) > 0:
                    return True
            return False
        else:
            return len(self.fixed_variables[name]) > 0

    def has_freed_variables(self, name=None):
        if name is None:
            for val in itervalues(self.freed_variables):
                if len(val) > 0:
                    return True
            return False
        else:
            return len(self.freed_variables[name]) > 0

    def clear_fixed_variables(self, name=None):
        if name is None:
            for key in self.fixed_variables:
                self.fixed_variables[key] = []
        else:
            if name in self.fixed_variables:
                self.fixed_variables[name] = []
            else:
                raise KeyError("KeyError: %s" % name)

    def clear_freed_variables(self, name=None):
        if name is None:
            for key in self.freed_variables:
                self.freed_variables[key] = []
        else:
            if name in self.freed_variables:
                self.freed_variables[name] = []
            else:
                raise KeyError("KeyError: %s" % name)

    #
    # Preprocess scenarios (ignoring bundles even if they exists)
    #

    def preprocess_scenarios(self, scenarios=None):

        start_time = time.time()

        if scenarios is None:
            scenarios = self._scenario_instance.keys()

        if self._options.verbose:
            print("Preprocessing %s scenarios" % len(scenarios))

        if self._options.verbose:
            if len(self._bundle_instances) > 0:
                print("Preprocessing scenarios without bundles. Bundle "
                      "preprocessing dependencies will be lost. Scenario "
                      "preprocessing flags must be reset before preprocessing "
                      "bundles.")

        for scenario_name in scenarios:

            self._preprocess_scenario(scenario_name,
                                      self._scenario_solver[scenario_name])

            # We've preprocessed the instance, reset the relevant flags
            self.clear_update_flags(scenario_name)
            self.clear_fixed_variables(scenario_name)
            self.clear_freed_variables(scenario_name)

        end_time = time.time()

        if self._options.output_times:
            print("Scenario preprocessing time=%.2f seconds"
                  % (end_time - start_time))

    #
    # Preprocess bundles (and the scenarios they depend on)
    #

    def preprocess_bundles(self,
                           bundles=None,
                           force_preprocess_bundle_objective=False,
                           force_preprocess_bundle_constraints=False):

        start_time = time.time()
        if len(self._bundle_instances) == 0:
            raise RuntimeError(
                "Unable to preprocess scenario bundles. Bundling "
                "does not seem to be activated.")

        if bundles is None:
            bundles = self._bundle_instances.keys()

        if self._options.verbose:
            print("Preprocessing %s bundles" % len(bundles))

        preprocess_bundle_objective = 0b01
        preprocess_bundle_constraints = 0b10

        for bundle_name in bundles:

            preprocess_bundle = 0
            solver = self._bundle_solvers[bundle_name]
            for scenario_name in self._bundle_scenarios[bundle_name]:

                if self.objective_updated[scenario_name]:
                    preprocess_bundle |= preprocess_bundle_objective
                if ((len(self.fixed_variables[scenario_name]) > 0) or \
                    (len(self.freed_variables[scenario_name]) > 0)) and \
                    self._options.preprocess_fixed_variables:
                    preprocess_bundle |= \
                        preprocess_bundle_objective | \
                        preprocess_bundle_constraints
                if self._bundle_first_preprocess[bundle_name]:
                    preprocess_bundle |= \
                        preprocess_bundle_objective | \
                        preprocess_bundle_constraints
                    self._bundle_first_preprocess[bundle_name] = False

                self._preprocess_scenario(scenario_name, solver)

                # We've preprocessed the instance, reset the relevant flags
                self.clear_update_flags(scenario_name)
                self.clear_fixed_variables(scenario_name)
                self.clear_freed_variables(scenario_name)

            if force_preprocess_bundle_objective:
                preprocess_bundle |= preprocess_bundle_objective
            if force_preprocess_bundle_constraints:
                preprocess_bundle |= preprocess_bundle_constraints

            if preprocess_bundle:

                bundle_ef_instance = \
                    self._bundle_instances[bundle_name]

                if solver.problem_format == ProblemFormat.nl:
                    idMap = {}
                    if preprocess_bundle & preprocess_bundle_objective:
                        ampl_preprocess_block_objectives(bundle_ef_instance,
                                                         idMap=idMap)
                    if preprocess_bundle & preprocess_bundle_constraints:
                        ampl_preprocess_block_constraints(bundle_ef_instance,
                                                          idMap=idMap)
                else:
                    idMap = {}
                    if preprocess_bundle & preprocess_bundle_objective:
                        canonical_preprocess_block_objectives(
                            bundle_ef_instance,
                            idMap=idMap)
                    if preprocess_bundle & preprocess_bundle_constraints:
                        canonical_preprocess_block_constraints(
                            bundle_ef_instance,
                            idMap=idMap)

        end_time = time.time()

        if self._options.output_times:
            print("Bundle preprocessing time=%.2f seconds"
                  % (end_time - start_time))

    def _preprocess_scenario(self, scenario_name, solver):

        assert scenario_name in self._scenario_instance
        scenario_objective_active = self._scenario_objective[scenario_name].active
        # because the preprocessor will skip the scenario objective if it is
        # part of a bundle and not active
        self._scenario_objective[scenario_name].activate()
        def _cleanup():
            if not scenario_objective_active:
                self._scenario_objective[scenario_name].deactivate()
        scenario_instance = self._scenario_instance[scenario_name]
        instance_fixed_variables = self.fixed_variables[scenario_name]
        instance_freed_variables = self.freed_variables[scenario_name]
        instance_all_constraints_updated = \
            self.all_constraints_updated[scenario_name]
        instance_constraints_updated_list = \
            self.constraints_updated_list[scenario_name]
        instance_objective_updated = self.objective_updated[scenario_name]

        persistent_solver_in_use = isinstance(solver, PersistentSolver)
        if (not instance_objective_updated) and \
           (not instance_fixed_variables) and \
           (not instance_freed_variables) and \
           (not instance_all_constraints_updated) and \
           (len(instance_constraints_updated_list) == 0):
            if persistent_solver_in_use:
                assert solver.instance_compiled()

            # instances are already preproccessed, nothing
            # needs to be done
            if self._options.verbose:
                print("No preprocessing necessary for scenario %s"
                      % (scenario_name))
            _cleanup()
            return

        if (instance_fixed_variables or instance_freed_variables) and \
           (self._options.preprocess_fixed_variables):

            if self._options.verbose:
                print("Running full preprocessing for scenario %s"
                      % (scenario_name))

            if solver.problem_format() == ProblemFormat.nl:
                ampl_expression_preprocessor({}, model=scenario_instance)
            else:
                canonical_expression_preprocessor({}, model=scenario_instance)

            # We've preprocessed the entire instance, no point in checking
            # anything else
            _cleanup()
            return

        if instance_objective_updated:

            if self._options.verbose:
                print("Preprocessing objective for scenario %s"
                      % (scenario_name))

            # if only the objective changed, there is minimal work to do.
            if solver.problem_format() == ProblemFormat.nl:
                ampl_preprocess_block_objectives(scenario_instance)
            else:
                canonical_preprocess_block_objectives(scenario_instance)

            if persistent_solver_in_use and \
               solver.instance_compiled():
                solver.compile_objective(scenario_instance)

        if (instance_fixed_variables or instance_freed_variables) and \
           (persistent_solver_in_use):

            if self._options.verbose:
                print("Compiling fixed status updates in persistent solver "
                      "for scenario %s" % (scenario_name))

            # it can be the case that the solver plugin no longer has an
            # instance compiled, depending on what state the solver plugin
            # is in relative to the instance.  if this is the case, just
            # don't compile the variable bounds.
            if solver.instance_compiled():
                variables_to_change = \
                    instance_fixed_variables + instance_freed_variables
                solver.compile_variable_bounds(
                    scenario_instance,
                    vars_to_update=variables_to_change)

        if instance_all_constraints_updated:

            if self._options.verbose:
                print("Preprocessing all constraints for scenario %s"
                      % (scenario_name))

            if solver.problem_format() == ProblemFormat.nl:
                idMap = {}
                for block in scenario_instance.block_data_objects(
                        active=True,
                        descend_into=True):
                    ampl_preprocess_block_constraints(block, idMap=idMap)
            else:
                idMap = {}
                for block in scenario_instance.block_data_objects(
                        active=True,
                        descend_into=True):
                    canonical_preprocess_block_constraints(block, idMap=idMap)

        elif len(instance_constraints_updated_list) > 0:

            # TODO
            assert not persistent_solver_in_use

            if self._options.verbose:
                print("Preprocessing constraint list (size=%s) for "
                      "scenario %s" % (len(instance_constraints_updated_list),
                                       scenario_name))

            idMap = {}
            repn_name = None
            repn_func = None
            if solver.problem_format() == ProblemFormat.nl:
                repn_name = "_ampl_repn"
                repn_func = generate_ampl_repn
            else:
                repn_name = "_canonical_repn"
                repn_func = generate_canonical_repn

            for constraint_data in instance_constraints_updated_list:
                if isinstance(constraint_data, LinearCanonicalRepn):
                    continue
                block = constraint_data.parent_block()
                # Get/Create the ComponentMap for the repn storage
                if not hasattr(block, repn_name):
                    setattr(block, repn_name, ComponentMap())
                getattr(block, repn_name)[constraint_data] = \
                    repn_func(constraint_data.body, idMap=idMap)

        if persistent_solver_in_use and \
           (not solver.instance_compiled()):
             solver.compile_instance(
                 scenario_instance,
                 symbolic_solver_labels=self._options.symbolic_solver_labels,
                 output_fixed_variable_bounds=not self._options.preprocess_fixed_variables)

        _cleanup()

    def get_solver_keywords(self):

        kwds = {}
        if not self._options.disable_advanced_preprocessing:
            if not self._options.preprocess_fixed_variables:
                kwds['output_fixed_variable_bounds'] = True

        return kwds
