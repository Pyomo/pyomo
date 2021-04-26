#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ("ScenarioTreePreprocessor",)

# TODO: Verify what needs to be done for persistent solver plugins
#       when advanced preprocessing is disabled and finish
#       implementing for that option.
import time

# these are the only two preprocessors currently invoked by the
# simple_preprocessor, which in turn is invoked by the preprocess()
# method of PyomoModel.
from pyomo.core.base.objective import Objective
from pyomo.core.base.constraint import Constraint
from pyomo.repn.standard_repn import (preprocess_block_objectives,
                                      preprocess_block_constraints,
                                      preprocess_constraint_data)

from pyomo.pysp.util.config import (PySPConfigBlock,
                                    safe_declare_common_option)
from pyomo.pysp.util.configured_object import PySPConfiguredObject

from six import itervalues

def _preprocess(model, objective=True, constraints=True):
    objective_found = False
    if objective:
        for block in model.block_data_objects(active=True):
            for obj in block.component_data_objects(Objective,
                                                    active=True,
                                                    descend_into=False):
                objective_found = True
                preprocess_block_objectives(block)
                break
            if objective_found:
                break
    if constraints:
        for block in model.block_data_objects(active=True):
            preprocess_block_constraints(block)


#
# We only want to do the minimal amount of work to get the instance
# back to a consistent "preprocessed" state. The following attributes
# are introduced to help perform the minimal amount of work, and
# should be augmented in the future if we can somehow do less. These
# attributes are initially cleared, and are re-set - following
# preprocessing, if necessary - before each round of model I/O.
#
class ScenarioTreePreprocessor(PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()

        safe_declare_common_option(options,
                                   "preprocess_fixed_variables")
        safe_declare_common_option(options,
                                   "symbolic_solver_labels")
        safe_declare_common_option(options,
                                   "output_times")
        safe_declare_common_option(options,
                                   "verbose")

        return options

    def __init__(self, *args, **kwds):

        super(ScenarioTreePreprocessor, self).__init__(*args, **kwds)
        self._scenario_solvers = {}
        self._scenario_instances = {}
        self._scenario_objectives = {}
        self._scenario_first_preprocess = {}

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
        self.constraints_updated_list = {}
        self.constraints_added_list = {}
        self.constraints_removed_list = {}

    def add_scenario(self, scenario, scenario_instance, scenario_solver):

        scenario_name = scenario.name
        assert scenario_name not in self._scenario_instances
        assert scenario_name not in self._scenario_to_bundle_map

        self._scenario_instances[scenario_name] = scenario_instance
        self._scenario_solvers[scenario_name] = scenario_solver
        self._scenario_objectives[scenario_name] = scenario._instance_objective
        self._scenario_first_preprocess[scenario_name] = True

        self.fixed_variables[scenario_name] = []
        self.freed_variables[scenario_name] = []
        self.objective_updated[scenario_name] = True
        self.constraints_updated_list[scenario_name] = []
        self.constraints_added_list[scenario_name] = []
        self.constraints_removed_list[scenario_name] = []

        scenario_instance = self._scenario_instances[scenario_name]
        assert scenario_instance is not None
        for block in scenario_instance.block_data_objects(active=True):
            assert not hasattr(block, "_gen_obj_repn")
            assert not hasattr(block, "_gen_con_repn")
            block._gen_obj_repn = False
            block._gen_con_repn = False

    def remove_scenario(self, scenario):
        scenario_name = scenario.name
        assert scenario_name in self._scenario_instances
        assert scenario_name not in self._scenario_to_bundle_map

        scenario_instance = self._scenario_instances[scenario_name]
        assert scenario_instance is not None
        for block in scenario_instance.block_data_objects(active=True):
            assert not block._gen_obj_repn
            assert not block._gen_con_repn
            del block._gen_obj_repn
            del block._gen_con_repn

        del self._scenario_instances[scenario_name]
        del self._scenario_solvers[scenario_name]
        del self._scenario_objectives[scenario_name]
        del self._scenario_first_preprocess[scenario_name]

        del self.fixed_variables[scenario_name]
        del self.freed_variables[scenario_name]
        del self.objective_updated[scenario_name]
        del self.constraints_updated_list[scenario_name]
        del self.constraints_added_list[scenario_name]
        del self.constraints_removed_list[scenario_name]

    def add_bundle(self, bundle, bundle_instance, bundle_solver):

        bundle_name = bundle.name
        assert bundle_name not in self._bundle_instances

        self._bundle_instances[bundle_name] = bundle_instance
        self._bundle_solvers[bundle_name] = bundle_solver
        self._bundle_scenarios[bundle_name] = list(bundle._scenario_names)
        self._bundle_first_preprocess[bundle_name] = True

        for scenario_name in self._bundle_scenarios[bundle_name]:
            assert scenario_name in self._scenario_instances
            assert scenario_name not in self._scenario_to_bundle_map
            self._scenario_to_bundle_map[scenario_name] = bundle_name

    def remove_bundle(self, bundle):

        bundle_name = bundle.name
        assert bundle_name in self._bundle_instances

        for scenario_name in self._bundle_scenarios[bundle_name]:
            assert scenario_name in self._scenario_instances
            assert scenario_name in self._scenario_to_bundle_map
            del self._scenario_to_bundle_map[scenario_name]

        del self._bundle_instances[bundle_name]
        del self._bundle_solvers[bundle_name]
        del self._bundle_scenarios[bundle_name]
        del self._bundle_first_preprocess[bundle_name]

    def clear_update_flags(self, name=None):
        if name is not None:
            self.objective_updated[name] = False
            self.constraints_updated_list[name] = []
            self.constraints_added_list[name] = []
            self.constraints_removed_list[name] = []
        else:
            for key in self.instances:
                self.objective_updated[key] = False
                self.constraints_updated_list[key] = []
                self.constraints_added_list[key] = []
                self.constraints_removed_list[key] = []

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
            scenarios = self._scenario_instances.keys()

        if self.get_option("verbose"):
            print("Preprocessing %s scenarios" % len(scenarios))

        if self.get_option("verbose"):
            if len(self._bundle_instances) > 0:
                print("Preprocessing scenarios without bundles. Bundle "
                      "preprocessing dependencies will be lost. Scenario "
                      "preprocessing flags must be reset before preprocessing "
                      "bundles.")

        for scenario_name in scenarios:

            self._preprocess_scenario(scenario_name,
                                      self._scenario_solvers[scenario_name])

            # We've preprocessed the instance, reset the relevant flags
            self._scenario_first_preprocess[scenario_name] = False
            self.clear_update_flags(scenario_name)
            self.clear_fixed_variables(scenario_name)
            self.clear_freed_variables(scenario_name)

        end_time = time.time()

        if self.get_option("output_times"):
            print("Scenario preprocessing time=%.2f seconds"
                  % (end_time - start_time))

    #
    # Preprocess bundles (and the scenarios they depend on)
    #

    def preprocess_bundles(self,
                           bundles=None,
                           force_preprocess_bundle_objective=False,
                           force_preprocess_bundle_constraints=False):
        # TODO: Does this import need to be delayed because
        #       it is in a plugins subdirectory
        from pyomo.solvers.plugins.solvers.persistent_solver import \
            PersistentSolver

        start_time = time.time()
        if len(self._bundle_instances) == 0:
            raise RuntimeError(
                "Unable to preprocess scenario bundles. Bundling "
                "does not seem to be activated.")

        if bundles is None:
            bundles = self._bundle_instances.keys()

        if self.get_option("verbose"):
            print("Preprocessing %s bundles" % len(bundles))

        preprocess_bundle_objective = 0b01
        preprocess_bundle_constraints = 0b10
        for bundle_name in bundles:

            preprocess_bundle = 0
            solver = self._bundle_solvers[bundle_name]
            persistent_solver_in_use = isinstance(solver, PersistentSolver)
            bundle_ef_instance = self._bundle_instances[bundle_name]
            if persistent_solver_in_use and \
               (not solver.has_instance()):
                assert self._bundle_first_preprocess[bundle_name]
                solver.set_instance(bundle_ef_instance)
                self._bundle_first_preprocess[bundle_name] = False
                for scenario_name in self._bundle_scenarios[bundle_name]:
                    self._scenario_solvers[scenario_name].set_instance(
                        self._scenario_instances[scenario_name])
                    # We've preprocessed the instance, reset the relevant flags
                    self._scenario_first_preprocess[scenario_name] = False
                    self.clear_update_flags(scenario_name)
                    self.clear_fixed_variables(scenario_name)
                    self.clear_freed_variables(scenario_name)
            else:
                if persistent_solver_in_use:
                    assert not self._bundle_first_preprocess[bundle_name]
                for scenario_name in self._bundle_scenarios[bundle_name]:
                    if self.objective_updated[scenario_name]:
                        preprocess_bundle |= preprocess_bundle_objective
                    if ((len(self.fixed_variables[scenario_name]) > 0) or \
                        (len(self.freed_variables[scenario_name]) > 0)) and \
                        self.get_option("preprocess_fixed_variables"):
                        preprocess_bundle |= \
                            preprocess_bundle_objective | \
                            preprocess_bundle_constraints
                    if self._bundle_first_preprocess[bundle_name]:
                        preprocess_bundle |= \
                            preprocess_bundle_objective | \
                            preprocess_bundle_constraints
                        self._bundle_first_preprocess[bundle_name] = False

                    if persistent_solver_in_use:
                        # also preprocess on the scenario solver
                        scenario_solver = self._scenario_solvers[scenario_name]
                        isinstance(scenario_solver, PersistentSolver)
                        self._preprocess_scenario(scenario_name,
                                                  scenario_solver)
                    self._preprocess_scenario(scenario_name, solver)

                    # We've preprocessed the instance, reset the relevant flags
                    self._scenario_first_preprocess[scenario_name] = False
                    self.clear_update_flags(scenario_name)
                    self.clear_fixed_variables(scenario_name)
                    self.clear_freed_variables(scenario_name)

                if force_preprocess_bundle_objective:
                    preprocess_bundle |= preprocess_bundle_objective

                if force_preprocess_bundle_constraints:
                    preprocess_bundle |= preprocess_bundle_constraints

                if preprocess_bundle:

                    if persistent_solver_in_use:
                        assert solver.has_instance()
                        if preprocess_bundle & preprocess_bundle_objective:
                            obj_count = 0
                            for obj in bundle_ef_instance.component_data_objects(
                                    ctype=Objective,
                                    descend_into=False,
                                    active=True):
                                obj_count += 1
                                if obj_count > 1:
                                    raise RuntimeError(
                                        "Persistent solver interface only "
                                        "supports a single active objective.")
                                solver.set_objective(obj)
                        if preprocess_bundle & preprocess_bundle_constraints:
                            # we assume the bundle constraints are just simple
                            # linking constraints (e.g., no SOSConstraints)
                            for con in bundle_ef_instance.component_data_objects(
                                    ctype=Constraint,
                                    descend_into=False,
                                    active=True):
                                solver.remove_constraint(con)
                                solver.add_constraint(con)
                    else:
                        idMap = {}
                        if preprocess_bundle & preprocess_bundle_objective:
                            preprocess_block_objectives(
                                bundle_ef_instance,
                                idMap=idMap)
                        if preprocess_bundle & preprocess_bundle_constraints:
                            preprocess_block_constraints(
                                bundle_ef_instance,
                                idMap=idMap)

        end_time = time.time()

        if self.get_option("output_times"):
            print("Bundle preprocessing time=%.2f seconds"
                  % (end_time - start_time))

    def _preprocess_scenario(self, scenario_name, solver):
        # TODO: Does this import need to be delayed because
        #       it is in a plugins subdirectory
        from pyomo.solvers.plugins.solvers.persistent_solver import \
            PersistentSolver

        assert scenario_name in self._scenario_instances
        scenario_objective_active = self._scenario_objectives[scenario_name].active
        # because the preprocessor will skip the scenario objective if it is
        # part of a bundle and not active
        self._scenario_objectives[scenario_name].activate()
        def _cleanup():
            if not scenario_objective_active:
                self._scenario_objectives[scenario_name].deactivate()
        scenario_instance = self._scenario_instances[scenario_name]
        instance_first_preprocess = self._scenario_first_preprocess[scenario_name]
        instance_fixed_variables = self.fixed_variables[scenario_name]
        instance_freed_variables = self.freed_variables[scenario_name]
        instance_constraints_updated_list = \
            self.constraints_updated_list[scenario_name]
        instance_constraints_added_list = \
            self.constraints_added_list[scenario_name]
        instance_constraints_removed_list = \
            self.constraints_removed_list[scenario_name]
        instance_objective_updated = self.objective_updated[scenario_name]

        persistent_solver_in_use = isinstance(solver, PersistentSolver)
        if (not instance_first_preprocess) and \
           (not instance_objective_updated) and \
           (not instance_fixed_variables) and \
           (not instance_freed_variables) and \
           (len(instance_constraints_updated_list) == 0) and \
           (len(instance_constraints_added_list) == 0) and \
           (len(instance_constraints_removed_list) == 0):
            if persistent_solver_in_use:
                assert solver.has_instance()

            # instances are already preproccessed, nothing
            # needs to be done
            if self.get_option("verbose"):
                print("No preprocessing necessary for scenario %s"
                      % (scenario_name))
            _cleanup()
            return

        if (not instance_first_preprocess) and \
           (instance_fixed_variables or instance_freed_variables):

            if persistent_solver_in_use:
                if solver.has_instance():
                    if self.get_option("verbose"):
                        print("Compiling fixed status updates in persistent solver "
                              "for scenario %s" % (scenario_name))

                    # it can be the case that the solver plugin no longer has an
                    # instance compiled, depending on what state the solver plugin
                    # is in relative to the instance.  if this is the case, just
                    # don't compile the variable bounds.
                    if solver.has_instance():
                        variables_to_change = \
                            instance_fixed_variables + instance_freed_variables
                        for var in variables_to_change:
                            solver.update_var(var)
            else:
                if self.get_option("preprocess_fixed_variables"):
                    if self.get_option("verbose"):
                        print("Running full preprocessing for scenario %s "
                              "due to fixing of variables"
                              % (scenario_name))
                    _preprocess(scenario_instance)
                    # We've preprocessed the entire instance, no point in checking
                    # anything else
                    _cleanup()
                    return

        if (not instance_first_preprocess) and \
           instance_objective_updated:

            if self.get_option("verbose"):
                print("Preprocessing objective for scenario %s"
                      % (scenario_name))

            if persistent_solver_in_use:
                if solver.has_instance():
                    obj_count = 0
                    for obj in scenario_instance.component_data_objects(
                            ctype=Objective,
                            descend_into=True,
                            active=True):
                        obj_count += 1
                        if obj_count > 1:
                            raise RuntimeError("Persistent solver interface only "
                                               "supports a single active objective.")
                        solver.set_objective(obj)
            else:
                # if only the objective changed, there is minimal work to do.
                _preprocess(scenario_instance,
                            objective=True,
                            constraints=False)

        if (not instance_first_preprocess) and \
           ((len(instance_constraints_updated_list) > 0) or \
            (len(instance_constraints_added_list) > 0) or \
            (len(instance_constraints_removed_list) > 0)):

            if persistent_solver_in_use:
                if solver.has_instance():
                    if self.get_option("verbose"):
                        print("Compiling constraint list (size=%s) for "
                              "scenario %s" % (scenario_name))
                    for con in instance_constraints_updated_list:
                        if (not con.has_lb()) and \
                           (not con.has_ub()):
                            assert not con.equality
                            continue  # non-binding, so skip
                        solver.remove_constraint(con)
                        solver.add_constraint(con)
                    for con in instance_constraints_removed_list:
                        if (not con.has_lb()) and \
                           (not con.has_ub()):
                            assert not con.equality
                            continue  # non-binding, so skip
                        solver.remove_constraint(con)
                    for con in instance_constraints_added_list:
                        if (not con.has_lb()) and \
                           (not con.has_ub()):
                            assert not con.equality
                            continue  # non-binding, so skip
                        solver.add_constraint(con)
            elif (len(instance_constraints_updated_list) > 0) or \
                 (len(instance_constraints_added_list) > 0):
                if self.get_option("verbose"):
                    print("Preprocessing constraint list (size=%s) for "
                          "scenario %s" % (len(instance_constraints_updated_list),
                                           scenario_name))
                idMap = {}
                for list_ in (instance_constraints_updated_list,
                              instance_constraints_added_list):
                    for constraint_data in list_:
                        if getattr(constraint_data, "_linear_canonical_form", False):
                            continue
                        preprocess_constraint_data(scenario_instance,
                                                   constraint_data,
                                                   idMap=idMap)

        if persistent_solver_in_use:
            if not solver.has_instance():
                solver.set_instance(
                    scenario_instance,
                    symbolic_solver_labels=\
                        self.get_option("symbolic_solver_labels"),
                    output_fixed_variable_bounds=\
                        not self.get_option("preprocess_fixed_variables"))
        elif instance_first_preprocess:
            if self.get_option("verbose"):
                print("Running initial full preprocessing for scenario %s"
                      % (scenario_name))
            _preprocess(scenario_instance)
        _cleanup()

    def modify_scenario_solver_keywords(self, scenario_name, kwds):
        # TODO: Does this import need to be delayed because
        #       it is in a plugins subdirectory
        from pyomo.solvers.plugins.solvers.persistent_solver import \
            PersistentSolver

        solver = self._scenario_solvers[scenario_name]
        if isinstance(solver, PersistentSolver):
            # these were applied when set_instance was called
            kwds.pop("output_fixed_variable_bounds",None)
            kwds.pop("symbolic_solver_labels",None)

        return kwds

    def modify_bundle_solver_keywords(self, bundle_name, kwds):
        # TODO: Does this import need to be delayed because
        #       it is in a plugins subdirectory
        from pyomo.solvers.plugins.solvers.persistent_solver import \
            PersistentSolver

        solver = self._bundle_solvers[bundle_name]

        if isinstance(solver, PersistentSolver):
            # these were applied when set_instance was called
            kwds.pop("output_fixed_variable_bounds",None)
            kwds.pop("symbolic_solver_labels",None)

        return kwds
