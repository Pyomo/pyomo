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
import copy

import pyomo.util.plugin
from pyomo.opt import SolverFactory, PersistentSolver
from pyomo.core import *
from pyomo.pysp import phextension
from pyomo.pysp.plugins.phboundextension import (_PHBoundBase,
                                                 ExtractInternalNodeSolutionsforInner)

logger = logging.getLogger('pyomo.pysp')

class convexhullboundextension(pyomo.util.plugin.SingletonPlugin, _PHBoundBase):

    pyomo.util.plugin.implements(phextension.IPHExtension)

    pyomo.util.plugin.alias("convexhullboundextension")

    def __init__(self, *args, **kwds):

        _PHBoundBase.__init__(self)

        # the bundle dual master model
        self._master_model = None

        # maps (iteration, scenario_name) to objective function value
        self._past_objective_values = {}

        # maps (iteration, scenario_name, variable_id) to value
        # maps iteration -> copy of each scenario solution
        self._past_var_values = {}

    def _iteration_k_bound_solves(self, ph, storage_key):

        # ** Code uses the values stored in the scenario solutions
        #    to perform a weighted vote in the case of discrete
        #    variables, so it is important that we execute this
        #    before perform any new subproblem solves.
        # candidate_sol is sometimes called xhat
        candidate_sol = ExtractInternalNodeSolutionsforInner(ph)
        # Caching the current set of ph solutions so we can restore
        # the original results. We modify the scenarios and re-solve -
        # which messes up the warm-start, which can seriously impact
        # the performance of PH. plus, we don't want lower bounding to
        # impact the primal PH in any way - it should be free of any
        # side effects.
        self.CachePHSolution(ph)

        # Save the current fixed state and fix queue
        self.RelaxPHFixedVariables(ph)

        # Assuming the weight terms are already active but proximal
        # terms need to be deactivated deactivate all proximal terms
        # and activate all weight terms
        self.DeactivatePHObjectiveProximalTerms(ph)

        # before we do anything, we need to cache the current PH
        # weights, so we can restore them below.
        self.CachePHWeights(ph)

        # for these solves, we're going to be using the convex hull
        # master problem weights.
        self._push_weights_to_ph(ph)

        # Weights have not been pushed to instance parameters (or
        # transmitted to the phsolverservers) at this point
        ph._push_w_to_instances()

        # assign rhos from ck.
        self._assign_cks(ph)

        failures = ph.solve_subproblems(warmstart=not ph._disable_warmstarts)

        if len(failures):

            print("Failed to compute duality-based bound due to "
                  "one or more solve failures")
            self._outer_bound_history[storage_key] = \
                float('-inf') if self._is_minimizing else float('inf')
            self._outer_status_history[storage_key] = self.STATUS_SOLVE_FAILED

        else:

            if ph._verbose:
                print("Successfully completed PH bound extension "
                      "iteration %s solves\n"
                      "- solution statistics:\n" % (storage_key))
                if ph._scenario_tree.contains_bundles():
                    ph.report_bundle_objectives()
                ph.report_scenario_objectives()

            # Compute the outer bound on the objective function.
            self._outer_bound_history[storage_key], \
                self._outer_status_history[storage_key] = \
                    self.ComputeOuterBound(ph, storage_key)

        # now change over to finding a feasible incumbent.
        print("Computing objective %s bound" %
              ("inner" if self._is_minimizing else "outer"))

        # push the updated outer bound to PH, for reporting purposes.
        ph._update_reported_bounds(self._outer_bound_history[storage_key])

        # Deactivate the weight terms.
        self.DeactivatePHObjectiveWeightTerms(ph)

        # Fix all non-leaf stage variables involved
        # in non-anticipativity conditions to the most
        # recently computed xbar (or something like it)
        self.FixScenarioTreeVariables(ph, candidate_sol)

        failures = ph.solve_subproblems(warmstart=not ph._disable_warmstarts,
                                        exception_on_failure=False)

        if len(failures):

            print("Failed to compute bound at xbar due to "
                  "one or more solve failures")
            self._inner_bound_history[storage_key] = \
                float('inf') if self._is_minimizing else float('-inf')
            self._inner_status_history[storage_key] = self.STATUS_SOLVE_FAILED

        else:

            if ph._verbose:
                print("Successfully completed PH bound extension "
                      "fixed-to-xbar solves for iteration %s\n"
                      "- solution statistics:\n" % (storage_key))
                if ph._scenario_tree.contains_bundles():
                    ph.report_bundle_objectives()
                ph.report_scenario_objectives()

            # Compute the inner bound on the objective function.
            self._inner_bound_history[storage_key], \
                self._inner_status_history[storage_key] = \
                    self.ComputeInnerBound(ph, storage_key)

        # push the updated inner bound to PH, for reporting purposes.
        ph._update_reported_bounds(inner = self._inner_bound_history[storage_key])

        # Restore ph to its state prior to entering this method (e.g.,
        # fixed variables, scenario solutions, proximal terms,
        # weights)
        self.RestorePH(ph)

    def _construct_bundle_dual_master_model(self, ph):

        self._master_model = ConcreteModel()
        for scenario in ph._scenario_tree._scenarios:
            for tree_node in scenario._node_list[:-1]:
                new_w_variable_name = "WVAR_"+str(tree_node._name)+"_"+str(scenario._name)
                new_w_k_parameter_name = "WDATA_"+str(tree_node._name)+"_"+str(scenario._name)+"_K"
                setattr(self._master_model,
                        new_w_variable_name,
                        Var(tree_node._standard_variable_ids,
                            within=Reals))
                setattr(self._master_model,
                        new_w_k_parameter_name,
                        Param(tree_node._standard_variable_ids,
                              within=Reals,
                              default=0.0,
                              mutable=True))
                setattr(self._master_model,
                        "V_"+str(scenario._name),
                        Var(within=Reals))
                # HERE - NEED TO MAKE CK VARAIBLE-DEPENDENT - PLUS WE NEED A SANE INITIAL VALUE (AND SUBSEQUENT VALUE)
        # DLW SAYS NO - THIS SHOULD BE VARIABLE-SPECIFIC
        setattr(self._master_model,
                "CK",
                Param(default=1.0, mutable=True))

        def obj_rule(m):
            expr = 0.0
            for scenario in ph._scenario_tree._scenarios:
                for tree_node in scenario._node_list[:-1]:
                    new_w_variable_name = "WVAR_"+str(tree_node._name)+"_"+str(scenario._name)
                    new_w_k_parameter_name = "WDATA_"+str(tree_node._name)+"_"+str(scenario._name)+"_K"
                    w_variable = m.find_component(new_w_variable_name)
                    w_k_parameter = m.find_component(new_w_k_parameter_name)
                    expr += 1.0/(2.0*m.CK) * sum(w_variable[i]**2 - 2.0*w_variable[i]*w_k_parameter[i] for i in w_variable)
                expr -= getattr(m, "V_"+str(scenario._name))
            return expr

        self._master_model.TheObjective = Objective(sense=minimize, rule=obj_rule)

        self._master_model.W_Balance = ConstraintList()

        for stage in ph._scenario_tree._stages[:-1]:

            for tree_node in stage._tree_nodes:

                # GABE SHOULD HAVE A SERVICE FOR THIS???
                for idx in tree_node._standard_variable_ids:

                    expr = 0.0
                    for scenario in tree_node._scenarios:
                        scenario_probability = scenario._probability
                        new_w_variable_name = "WVAR_"+str(tree_node._name)+"_"+str(scenario._name)
                        w_variable = self._master_model.find_component(new_w_variable_name)
                        expr += scenario_probability * w_variable[idx]

                    self._master_model.W_Balance.add(expr == 0.0)

        # we can't populate until we see data from PH....
        self._master_model.V_Bound = ConstraintList()

#        self._master_model.pprint()

    #
    # populate the master bundle model from the PH parameters
    #
    def _populate_bundle_dual_master_model(self, ph):

        current_iteration = ph._current_iteration

        # first step is to update the historical information from PH

        for scenario in ph._scenario_tree._scenarios:
            primal_objective_value = scenario._objective
            self._past_objective_values[(current_iteration, scenario._name)] = primal_objective_value

#        print "PAST OBJECTIVE FUNCTION VALUES=",self._past_objective_values

        assert current_iteration not in self._past_var_values
        iter_var_values = self._past_var_values[current_iteration] = {}
        for scenario in ph._scenario_tree._scenarios:
            iter_var_values[scenario._name] = copy.deepcopy(scenario._x)

#        print "PAST VAR VALUES=",self._past_var_values

        # propagate PH parameters to concrete model and re-preprocess.
        for scenario in ph._scenario_tree._scenarios:
            for tree_node in scenario._node_list[:-1]:
                new_w_k_parameter_name = \
                    "WDATA_"+str(tree_node._name)+"_"+str(scenario._name)+"_K"
                w_k_parameter = \
                    self._master_model.find_component(new_w_k_parameter_name)
                ph_weights = scenario._w[tree_node._name]

                for idx in w_k_parameter:
                    w_k_parameter[idx] = ph_weights[idx]

        # V bounds are per-variable, per-iteration
        for scenario in ph._scenario_tree._scenarios:
            scenario_name = scenario._name
            v_var = getattr(self._master_model, "V_"+str(scenario_name))
            expr = self._past_objective_values[(current_iteration, scenario_name)]
            for tree_node in scenario._node_list[:-1]:
                new_w_variable_name = "WVAR_"+str(tree_node._name)+"_"+str(scenario_name)
                w_variable = self._master_model.find_component(new_w_variable_name)
                expr += sum(iter_var_values[scenario_name][tree_node._name][var_id] * w_variable[var_id] for var_id in w_variable)

            self._master_model.V_Bound.add(v_var <= expr)

#        print "V_BOUNDS CONSTRAINT:"
#        self._master_model.V_Bound.pprint()
        with SolverFactory(ph._solver_type,
                           solver_io=ph._solver_io) as solver:
            # the reason we go through this trouble rather
            # than harcoding cplex as the solver is so that
            # we can test the script and not have to worry
            # about a missing solver
            if isinstance(solver, PersistentSolver):
                solver.compile_instance(self._master_model)
            results = solver.solve(self._master_model)
            self._master_model.solutions.load_from(results)
#        print "MASTER MODEL WVAR FOLLOWING SOLVE:"
#        self._master_model.pprint()

#        self._master_model.pprint()

    #
    # take the weights from the current convex hull master problem
    # solution, and push them into the PH scenario instances - so we
    # can do the solves and compute a lower bound.
    #

    def _push_weights_to_ph(self, ph):

        for scenario in ph._scenario_tree._scenarios:
            for tree_node in scenario._node_list[:-1]:

                new_w_variable_name = "WVAR_"+str(tree_node._name)+"_"+str(scenario._name)
                w_variable = self._master_model.find_component(new_w_variable_name)

                ph_weights = scenario._w[tree_node._name]

                for idx in w_variable:
                    ph_weights[idx] = w_variable[idx].value

    #
    # move the variable rhos from PH into the analogous CK parameter
    # in the convex hull master.
    #
    def _assign_cks(self, ph):

        # TBD: for now, we're just grabbing the default rho from PH - we need to
        #      extract them per-variable in the very near future.
        self._master_model.CK = ph._rho

    ############ Begin Callback Functions ##############

    def reset(self, ph):
        self.__init__()

    def pre_ph_initialization(self,ph):
        """
        Called before PH initialization.
        """
        pass

    def post_instance_creation(self, ph):
        """
        Called after PH initialization has created the scenario
        instances, but before any PH-related
        weights/variables/parameters/etc are defined!
        """
        pass

    def post_ph_initialization(self, ph):
        """
        Called after PH initialization!
        """

        if ph._verbose:
            print("Invoking post initialization callback "
                  "in convexhullboundextension")

        self._is_minimizing = True if (ph._objective_sense == minimize) else False
        # TODO: Check for ph options that may not be compatible with
        #       this plugin and warn / raise exception

        # grab the update interval from the environment variable, if
        # it exists.
        update_interval_variable_name = "PHBOUNDINTERVAL"
        if update_interval_variable_name in os.environ:
            self._update_interval = int(os.environ[update_interval_variable_name])
            print("convexhullboundextension using update interval="
                  +str(self._update_interval)+", extracted from "
                  "environment variable="+update_interval_variable_name)
        else:
            print("convexhullboundextension using default update "
                  "interval="+str(self._update_interval))

        self._construct_bundle_dual_master_model(ph)

    def post_iteration_0_solves(self, ph):
        """Called after the iteration 0 solves!"""

        if ph._verbose:
            print("Invoking post iteration 0 solve callback "
                  "in convexhullboundextension")

        if ph._ph_warmstarted:
            print("PH warmstart detected. Bound computation requires solves "
                  "after iteration 0.")
            self.pre_iteration_k_solves(ph)
            return

        # Always compute a lower/upper bound here because it requires
        # no work.  The instances (or bundles) have already been
        # solved with the original (non-PH-augmented) objective and
        # are loaded with results.

        #
        # Note: We will still obtain a bound using the weights
        #       computed from PH iteration 0 in the
        #       pre_iteration_k_solves callback.
        #
        ph_iter = None

        # Note: It is important that the mipgap is not adjusted
        #       between the time after the subproblem solves
        #       and before now.
        self._outer_bound_history[ph_iter], \
            self._outer_status_history[ph_iter] = \
                self.ComputeOuterBound(ph, ph_iter)
        ph._update_reported_bounds(outer = self._outer_bound_history[ph_iter]) # dlw May 2016

        # YIKES - WHY IS THIS HERE????!!
        self._populate_bundle_dual_master_model(ph)

    def post_iteration_0(self, ph):
        """
        Called after the iteration 0 solves, averages computation, and
        weight computation!
        """
        pass

    def pre_iteration_k_solves(self, ph):
        """
        Called immediately before the iteration k solves!
        """

        if ph._verbose:
            print("Invoking pre iteration k solve callback "
                  "in convexhullboundextension")

        #
        # Note: We invoke this callback pre iteration k in order to
        #       obtain a PH bound using weights computed from the
        #       PREVIOUS iteration's scenario solutions (including
        #       those of iteration zero).
        #
        ph_iter = ph._current_iteration-1

        if (ph_iter % self._update_interval) != 0:
            return

        self._iteration_k_bound_solves(ph, ph_iter)

        # YIKES - WHY IS THIS HERE????!!
        self._populate_bundle_dual_master_model(ph)

        if ph._current_iteration > 5:
            print("WE ARE PAST ITERATION 5 - STARTING TO TRUST CONVEX "
                  "HULL BOUND EXTENSION FOR WEIGHT UPDATES")
            ph._ph_weight_updated_enabled = False
            self._push_weights_to_ph(ph)

    def post_iteration_k_solves(self, ph):
        """
        Called after the iteration k solves!
        """
        pass

    def post_iteration_k(self, ph):
        """
        Called after the iteration k is finished, after weights have
        been updated!
        """
        pass

    def post_ph_execution(self, ph):
        """
        Called after PH has terminated!
        """

        if ph._verbose:
            print("Invoking post execution callback in convexhullboundextension")

        #
        # Note: We invoke this callback in order to compute a bound
        #       using the weights obtained from the final PH
        #       iteration.
        #
        ph_iter = ph._current_iteration

        if (ph_iter % self._update_interval) == 0:

            self._iteration_k_bound_solves(ph, ph_iter)

        self.ReportBoundHistory()
        self.ReportBestBound()
