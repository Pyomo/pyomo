# May 2015 issue: reals really should not be *fixed* to compute an inner bound
# It seems like a lot to change their bounds, but that is probably the way to go.

from __future__ import division
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
from pyomo.pysp import phextension
from pyomo.core.base import minimize
from pyomo.opt import UndefinedData

from operator import itemgetter
from six import iteritems

from pyomo.pysp.phboundbase import (_PHBoundBase,
                                    ExtractInternalNodeSolutionsforInner)

class _PHBoundExtensionImpl(_PHBoundBase):

    def __init__(self):

        _PHBoundBase.__init__(self)

    def _iteration_k_bound_solves(self, ph, storage_key):

        # storage key is for results (e.g. the ph iter number)

        # ** Some code might use the values stored in the scenario solutions
        #    to perform a weighted vote in the case of discrete
        #    variables, so it is important that we execute this
        #    before perform any new subproblem solves.

        # candidate_sol is sometimes called xhat
        try:
            candidate_sol = ExtractInternalNodeSolutionsforInner(ph)
        except:
            print("Failed to extract candiate xhat for "
                  "inner bound computation using xhat_method %s. "
                  "Skipping inner bound computation."
                  % (ph._xhat_method))
            candidate_sol = None

        # Caching the current set of ph solutions so we can restore
        # the original results. We modify the scenarios and re-solve -
        # which messes up the warm-start, which can seriously impact
        # the performance of PH. plus, we don't want lower bounding to
        # impact the primal PH in any way - it should be free of any
        # side effects.
        self.CachePHSolution(ph)

        # Save the current fixed state and fix queue.
        self.RelaxPHFixedVariables(ph)

        # Assuming the weight terms are already active but proximal
        # terms need to be deactivated deactivate all proximal terms
        # and activate all weight terms.
        self.DeactivatePHObjectiveProximalTerms(ph)

        if candidate_sol is not None:
            # Deactivate the weight terms.
            self.DeactivatePHObjectiveWeightTerms(ph)

            # Fix all non-leaf stage variables involved
            # in non-anticipativity conditions to the most
            # recently computed xbar (or something like it)
            # integers should be truly fixed, but reals require special care
            self.FixScenarioTreeVariables(ph, candidate_sol)

            # now change over to finding a feasible incumbent.
            if ph._verbose:
                print("Computing objective %s bound" %
                      ("inner" if self._is_minimizing else "outer"))

            failures = ph.solve_subproblems(warmstart=not ph._disable_warmstarts,
                                            exception_on_failure=False)

            if len(failures):

                print("Failed to compute %s bound at xhat due to "
                      "one or more solve failures" %
                  ("inner" if self._is_minimizing else "outer"))
                self._inner_bound_history[storage_key] = \
                    float('inf') if self._is_minimizing else float('-inf')
                self._inner_status_history[storage_key] = self.STATUS_SOLVE_FAILED

            else:

                if ph._verbose:
                    print("Successfully completed PH bound extension "
                          "fixed-to-xhat solves for iteration %s\n"
                          "- solution statistics:\n" % (storage_key))
                    if ph._scenario_tree.contains_bundles():
                        ph.report_bundle_objectives()
                    ph.report_scenario_objectives()

                # Compute the inner bound on the objective function.
                IBval, IBstatus = self.ComputeInnerBound(ph, storage_key)
                self._inner_bound_history[storage_key] = IBval
                self._inner_status_history[storage_key] = IBstatus

            # Undo FixScenarioTreeVariables
            self.RestoreLastPHChange(ph)

            # Undo DeactivatePHObjectiveWeightTerms
            self.RestoreLastPHChange(ph)

        else:

            if self._is_minimizing:
                self._inner_bound_history[storage_key] = float('inf')
            else:
                self._inner_bound_history[storage_key] = float('-inf')
            self._inner_status_history[storage_key] = self.STATUS_NONE

        # push the updated inner bound to PH, for reporting purposes.
        ph._update_reported_bounds(inner = self._inner_bound_history[storage_key])

        # It is possible weights have not been pushed to instance
        # parameters (or transmitted to the phsolverservers) at this
        # point.
        ph._push_w_to_instances()

        failures = ph.solve_subproblems(warmstart=not ph._disable_warmstarts,
                                        exception_on_failure=False)

        if len(failures):

            print("Failed to compute duality-based bound due to "
                  "one or more solve failures")
            self._outer_bound_history[storage_key] = \
                float('-inf') if self._is_minimizing else float('inf')
            self._outer_status_history[storage_key] = self.STATUS_SOLVE_FAILED

        else:

            if ph._verbose:
                print("Successfully completed PH bound extension "
                      "weight-term only solves for iteration %s\n"
                      "- solution statistics:\n" % (storage_key))
                if ph._scenario_tree.contains_bundles():
                    ph.report_bundle_objectives()
                ph.report_scenario_objectives()

            # Compute the outer bound on the objective function.
            self._outer_bound_history[storage_key], \
                self._outer_status_history[storage_key] = \
                    self.ComputeOuterBound(ph, storage_key)

            ph._update_reported_bounds(outer = self._outer_bound_history[storage_key])

        # Restore ph to its state prior to entering this method (e.g.,
        # fixed variables, scenario solutions, proximal terms)
        self.RestorePH(ph)

    ############ Begin Callback Functions ##############

    def reset(self, ph):
        """Invoked to reset the state of a plugin to that of post-construction"""
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
        Called after PH initialization
        """

        if ph._verbose:
            print("Invoking post initialization callback "
                  "in phboundextension")

        self._is_minimizing = True if (ph._objective_sense == minimize) else False
        # TODO: Check for ph options that may not be compatible with
        #       this plugin and warn / raise exception

        # grab the update interval from the environment variable, if
        # it exists.
        update_interval_variable_name = "PHBOUNDINTERVAL"
        update_interval_file_name = "PHB_.DAT"
        if os.path.isfile(update_interval_file_name):
            print("phboundextension is getting the update interval from file=",
                   update_interval_file_name)
            with open(update_interval_file_name) as ifile:
                ifileval = ifile.read()
            if isinstance(ifileval, int):
                print ("update interval=",ifileval)
                self._update_interval = ifileval
            else:
                raise RuntimeError("The value must be of type integer, but the value read="+str(ifileval))

        elif update_interval_variable_name in os.environ:
            self._update_interval = int(os.environ[update_interval_variable_name])
            print("phboundextension using update interval="
                  +str(self._update_interval)+", extracted from "
                  "environment variable="+update_interval_variable_name)
        else:
            print("phboundextension using default update "
                  "interval="+str(self._update_interval))

    def post_iteration_0_solves(self, ph):
        """
        Called after the iteration 0 solves
        """

        if ph._verbose:
            print("Invoking post iteration 0 solve callback "
                  "in phboundextension")

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

        # dlw May 2016: the reported bound gets set for general iterations right after
        #               assignment to the history, so we do it here also
        ph._update_reported_bounds(outer = self._outer_bound_history[ph_iter])

    def post_iteration_0(self, ph):
        """
        Called after the iteration 0 solves, averages computation, and weight computation
        """
        pass

    def pre_iteration_k_solves(self, ph):
        """
        Called immediately before the iteration k solves
        """

        if ph._verbose:
            print("Invoking pre iteration k solve callback "
                  "in phboundextension")

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

    def post_iteration_k_solves(self, ph):
        """
        Called after the iteration k solves!
        """
        pass

    def post_iteration_k(self, ph):
        """
        Called after the iteration k is finished, after weights have been updated!
        """
        pass

    def post_ph_execution(self, ph):
        """
        Called after PH has terminated!
        """

        if ph._verbose:
            print("Invoking post execution callback in phboundextension")

        #
        # Note: We invoke this callback in order to compute a bound
        #       using the weights obtained from the final PH
        #       iteration.
        #
        ph_iter = ph._current_iteration
        self._iteration_k_bound_solves(ph, ph_iter)

        self.ReportBoundHistory()
        self.ReportBestBound()

class phboundextension(pyomo.util.plugin.SingletonPlugin, _PHBoundExtensionImpl):

    pyomo.util.plugin.implements(phextension.IPHExtension)

    pyomo.util.plugin.alias("phboundextension")

    def __init__(self):

        _PHBoundExtensionImpl.__init__(self)
