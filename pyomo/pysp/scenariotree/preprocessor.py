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

from collections import defaultdict

# these are the only two preprocessors currently invoked by the
# simple_preprocessor, which in turn is invoked by the preprocess()
# method of PyomoModel.
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
class ScenarioTreePreprocessor(object):

    @staticmethod
    def register_options(options):
        safe_register_common_option(options,
                                    "disable_advanced_preprocessing")
        safe_register_common_option(options,
                                    "preprocess_fixed_variables")

        #
        # various
        #
        safe_register_common_option(options,
                                    "output_times")
        safe_register_common_option(options,
                                    "verbose")

    def __init__(self,
                 options,
                 solver,
                 instances,
                 bundle_binding_instance_map=None,
                 bundle_scenario_instance_map=None):

        self._first = True
        self._options = options
        self._solver = solver
        self._instances = dict(instances)

        #
        # Bundle related objects
        #
        self._bundle_binding_instance_map = {}
        self._bundle_scenario_instance_map = {}
        self._scenario_to_bundle_map = {}
        self._bundle_first_preprocess = {}

        # maps between instance name and a list of
        # variable (name, index) pairs
        self.fixed_variables = dict((name,[]) for name in instances)
        self.freed_variables = dict((name,[]) for name in instances)
        # indicates update status of instances since the last
        # preprocessing round
        self.objective_updated = dict.fromkeys(instances, False)
        self.all_constraints_updated = dict.fromkeys(instances, False)
        self.constraints_updated_list = dict((name,[]) for name in instances)

        for scenario_name in self._instances:

            self.objective_updated[scenario_name] = True
            self.all_constraints_updated[scenario_name] = True

            if not self._options.disable_advanced_preprocessing:
                instance = self._instances[scenario_name]
                for block in instance.block_data_objects(active=True):
                    block._gen_obj_ampl_repn = False
                    block._gen_con_ampl_repn = False
                    block._gen_obj_canonical_repn = False
                    block._gen_con_canonical_repn = False

        #
        # validate bundles and create reverse map
        #
        if bundle_binding_instance_map is None:
            assert bundle_scenario_instance_map is None
        else:

            self._bundle_binding_instance_map = \
                dict(bundle_binding_instance_map)
            assert bundle_scenario_instance_map is not None
            self._bundle_scenario_instance_map = \
                dict(bundle_scenario_instance_map)
            self._bundle_first_preprocess = \
                dict.fromkeys(self._bundle_binding_instance_map, True)

            assert (sorted(self._bundle_binding_instance_map.keys()) == \
                    sorted(self._bundle_scenario_instance_map.keys()))
            bundle_scenarios = []
            for bundle_name in self._bundle_scenario_instance_map:
                for scenario_name in \
                       self._bundle_scenario_instance_map[bundle_name]:
                    self._scenario_to_bundle_map[scenario_name] = bundle_name
                    bundle_scenarios.append(scenario_name)
            assert (sorted(self._instances.keys()) == \
                    sorted(bundle_scenarios))

    def clear_update_flags(self, name=None):
        if name is not None:
            self.objective_updated[name] = False
            self.all_constraints_updated[name] = False
            self.constraints_updated_list[name] = []
        else:
            for key in self.instances:
                self.objective_updated[key] = False
                self.all_constraints_updated[key] = False
                self.user_constraints_updated[key] = []

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
    # Preprocesses subproblems. This detects whether or not bundling
    # is present and preprocesses accordingly. The optional
    # subproblems keyword should be assigned a list of bundle names if
    # bundling is active, otherwise a list of scenario names
    #

    def preprocess_subproblems(self, subproblems=None):

        if self._bundle_binding_instance_map is None:
            self.preprocess_scenarios(scenarios=subproblems)
        else:
            self.preprocess_bundles(bundles=subproblems)

    #
    # Preprocess scenarios (ignoring bundles even if they exists)
    #

    def preprocess_scenarios(self, scenarios=None):

        start_time = time.time()

        if scenarios is None:
            scenarios = self._instances.keys()

        if self._options.verbose:
            print("Preprocessing %s scenarios" % len(scenarios))

        if self._bundle_binding_instance_map is not None:
            print("Preprocessing scenarios without bundles. Bundle preprocessing "
                  "dependencies will be lost. Scenario preprocessing flags must "
                  "be reset before preprocessing bundles.")

        for scenario_name in scenarios:

            self._preprocess_scenario(scenario_name)

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

        if self._bundle_binding_instance_map is None:
            raise RuntimeError(
                "Unable to preprocess scenario bundles. Bundling "
                "does not seem to be activated.")

        if bundles is None:
            bundles = self._bundle_binding_instance_map.keys()

        if self._options.verbose:
            print("Preprocessing %s bundles" % len(bundles))

        preprocess_bundle_objective = 0b01
        preprocess_bundle_constraints = 0b10

        for bundle_name in bundles:

            preprocess_bundle = 0
            for scenario_name in self._bundle_scenario_instance_map[bundle_name]:

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

                self._preprocess_scenario(scenario_name)

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
                    self._bundle_binding_instance_map[bundle_name]

                if self._solver.problem_format == ProblemFormat.nl:
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

    def _preprocess_scenario(self, scenario_name):

        assert scenario_name in self._instances
        scenario_instance = self._instances[scenario_name]
        instance_fixed_variables = self.fixed_variables[scenario_name]
        instance_freed_variables = self.freed_variables[scenario_name]
        instance_all_constraints_updated = \
            self.all_constraints_updated[scenario_name]
        instance_constraints_updated_list = \
            self.constraints_updated_list[scenario_name]
        instance_objective_updated = self.objective_updated[scenario_name]

        persistent_solver_in_use = isinstance(self._solver, PersistentSolver)
        if (not instance_objective_updated) and \
           (not instance_fixed_variables) and \
           (not instance_freed_variables) and \
           (not instance_all_constraints_updated) and \
           (len(instance_constraints_updated_list) == 0):

            # instances are already preproccessed, nothing
            # needs to be done
            if self._options.verbose:
                print("No preprocessing necessary for scenario %s"
                      % (scenario_name))
            return

        if (instance_fixed_variables or instance_freed_variables) and \
           (self._options.preprocess_fixed_variables):

            if self._options.verbose:
                print("Running full preprocessing for scenario %s"
                      % (scenario_name))

            if self._solver.problem_format() == ProblemFormat.nl:
                ampl_expression_preprocessor({}, model=scenario_instance)
            else:
                canonical_expression_preprocessor({}, model=scenario_instance)

            # We've preprocessed the entire instance, no point in checking
            # anything else
            return

        if instance_objective_updated:

            if self._options.verbose:
                print("Preprocessing objective for scenario %s"
                      % (scenario_name))

            # if only the objective changed, there is minimal work to do.
            if self._solver.problem_format() == ProblemFormat.nl:
                ampl_preprocess_block_objectives(scenario_instance)
            else:
                canonical_preprocess_block_objectives(scenario_instance)

            if persistent_solver_in_use and self._solver.instance_compiled():
                self._solver.compile_objective(scenario_instance)

        if (instance_fixed_variables or instance_freed_variables) and \
           (persistent_solver_in_use):

            if self._options.verbose:
                print("Compiling fixed status updates in persistent solver "
                      "for scenario %s" % (scenario_name))

            # it can be the case that the solver plugin no longer has an
            # instance compiled, depending on what state the solver plugin
            # is in relative to the instance.  if this is the case, just
            # don't compile the variable bounds.
            if self._solver.instance_compiled():
                variables_to_change = \
                    instance_fixed_variables + instance_freed_variables
                self._solver.compile_variable_bounds(
                    scenario_instance,
                    vars_to_update=variables_to_change)

        if instance_all_constraints_updated:

            if self._options.verbose:
                print("Preprocessing all constraints for scenario %s"
                      % (scenario_name))

            if self._solver.problem_format() == ProblemFormat.nl:
                idMap = {}
                for block in scenario_instance.block_data_objects(active=True,
                                                                  descend_into=True):
                    ampl_preprocess_block_constraints(block, idMap=idMap)
            else:
                idMap = {}
                for block in scenario_instance.block_data_objects(active=True,
                                                                  descend_into=True):
                    canonical_preprocess_block_constraints(block, idMap=idMap)

        elif len(instance_constraints_updated_list) > 0:

            if self._options.verbose:
                print("Preprocessing constraint list (size=%s) for "
                      "scenario %s" % (len(instance_constraints_updated_list),
                                       scenario_name))

            idMap = {}
            repn_name = None
            repn_func = None
            if self._solver.problem_format() == ProblemFormat.nl:
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

    def get_solver_keywords(self):

        kwds = {}
        if self._options.disable_advanced_preprocesssing:
            if not self._options.preprocess_fixed_variables:
                kwds['output_fixed_variable_bounds'] = True

        return kwds
