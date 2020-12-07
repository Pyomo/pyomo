# DELETE THIS COMMENT: we need control over getting sorgw to run first
# brancher.py
# find a branching var for bbph
# pickle a special little object
# note: it could be "n/a"
# As of Sept 2016 this is heavily biased toward binaries

outputfilename = "brancherout.p"

import sys
import time
import os
import pickle # so we can send index across with its type (I hope)

import pyomo.common.plugin
from pyomo.core import *
from pyomo.pysp import phextension
from pyomo.pysp.phutils import *
from pyomo.pysp.generators import \
    scenario_tree_node_variables_generator_noinstances
import pyomo.solvers.plugins.smanager.phpyro
from pyomo.pysp import phsolverserverutils

thisfile = os.path.abspath(__file__)

#
# Create a dictionary mapping scenario tree id to
# variable bounds
#
def collect_node_variable_bounds(tree_node):
    assert tree_node._variable_datas is not None
    var_bounds = {}
    for variable_id, vardatas in iteritems(tree_node._variable_datas):
        vardata = vardatas[0][0]
        if not vardata.is_expression_type():
            var_bounds[variable_id] = (vardata.lb, vardata.ub)
        else:
            # Expression
            var_bounds[variable_id] = (None,None)

    return var_bounds

# This function will be executed on a phsolverserver using
# the transmit_external_function_invocation_to_worker utility
def external_collect_variable_bounds(ph,
                                     scenario_tree,
                                     scenario_or_bundle,
                                     node_list):

    variable_bounds = {}
    for node_name in node_list:
        tree_node = scenario_tree._tree_node_map[node_name]
        variable_bounds[node_name] = collect_node_variable_bounds(tree_node)

    return variable_bounds


#==================================================
class brancherextension(pyomo.common.plugin.SingletonPlugin):

    pyomo.common.plugin.implements(phextension.IPHExtension)
    
    # the below is a hack to get this extension into the
    # set of IPHExtension objects, so it can be queried
    # automagically by PH.
    pyomo.common.plugin.alias("brancherextension")

    def __init__(self, *args, **kwds):

        self.Tol = 1e-6

#==================================================
    def reset(self, ph):
        self.__init__()

    def pre_ph_initialization(self,ph):
        # we don't need to intefere with PH initialization.
        pass
#==================================================

#==================================================
    def post_instance_creation(self, ph):
        # we don't muck with the instances.
        pass

#==================================================
    def post_ph_initialization(self, ph):
        pass

#==================================================
    def post_iteration_0_solves(self, ph):
        pass

#==================================================
    def post_iteration_0(self, ph):
        pass        

#==================================================
    def pre_iteration_k_solves(self, ph):
        pass

#==================================================
    def post_iteration_k_solves(self, ph):
        pass

#==================================================
    def post_iteration_k(self, ph):
        pass

#==================================================
    def post_ph_execution(self, ph):

        ####################################################
        # copied from the archived, serial version of BBPH
        # find a var to branch on and pickle some info about it
        # As of Sept 2016, this is getting pretty hacked up
        

        start_time = time.time()
        gotit, varpair = self.Select_var_by_W_dispersion(ph)
        if gotit:
            source = "sorgw"
        else:
            dist, stagenum, varpair = self.Select_var_by_dist_from_xbar(ph)
            if stagenum is not None:
                source = "Based on dist from xbar"
            else:
                # All at xbar, so trying latest converged
                latestSN, (latestTN, latestVI), couldlatest = self.Select_var_by_latest_convergence(ph)
                source = "Latest Convergence"
                if not couldlatest:
                    # print ("resorting to unbranched")
                    # use the vars with "latest" names even though that is not the correct name...
                    latestSN, latestTN, latestVI = self.Select_var_by_first_unbranched(ph)
                if latestSN is None:  # we struck out...
                        xfer = ("N/A","N/A","N/A","FAILED TO FIND A BRANCHING VAR")
                        pickle.dump( xfer, open( outputfilename, "wb" ) )
                        print ("Time in brancher.py=",time.time()-start_time)
                        return
                varpair = (latestTN, latestVI)

        node_name, variable_id = varpair
        tree_node = ph._scenario_tree._tree_node_map[node_name]
        varname, varindex = tree_node._variable_ids[variable_id]
        xfer = (node_name, varname, varindex, source)
        pickle.dump( xfer, open( outputfilename, "wb" ) )

        print ("Time in brancher.py=",time.time()-start_time)

#=========================

    def Select_var_by_W_dispersion(self, ph):
        # assuming the solve is complete, find a good variable on which to branch
        # Sept 2016
        # Per a suggestion by John S, maybe we want a var where W is highly dispersed, but the var still didn't converge
        #   but I'm not sure how to measure the dispersion in an appropriate way,
        # I think what we really want is a var where W behaved badly (in the sorgw.py sense)
        #   or a var that cycled (but unlike WW, we want the "most expensive" var that cycled)
        # Wait! Now that I think about it, maybe max Range (as suggested by John) is an OK proxy (it will also be influenced a lot by rho)
        #   for "most expensive."
        # Idea (Sept 2016, DLW):
        #  Read the sorgw interesting variables file
        #  you need to lowest stage with a non-converged var whether it is in the file or not
        #  so first, find that stage
        #  Loop through the sorgw vars and keep only those that are not converged and from that stage
        #  from among those, run a contest to find the highest W range
        #  .... Let's let sorgw.py do this work by sticking something on ph
        # ASSUME sorgw was root node only !!!
        root_node_name = ph._scenario_tree.findRootNode()._name

        if hasattr(ph, "_sorgw_BiggestLoser"):
            print ("sorgw_BiggestLoser="+str(ph._sorgw_BiggestLoser))
            return True, (root_node_name, ph._sorgw_BiggestLoser)
        else:
            return False, ("N/A", 0)
        
#=========================
    def Select_var_by_dist_from_xbar(self, ph):
        # assuming the solve is complete, find a good variable on which to branch
        # we want the biggest dist from the lowest stage with a non-zero...
        # this is looking for an outlier with general ints, but for binaries, it picks the var
        # with xbar closest to 1/2
        biggestdist = 0
        stagenum = 0
        # TBD: we probably should go all the way out, as in the paper...
        for stage in ph._scenario_tree._stages[:-1]:
            stagenum += 1
            for tree_node in stage._tree_nodes:
                xbars = tree_node._xbars
                mins = tree_node._minimums
                maxs = tree_node._maximums
                for scen in tree_node._scenarios:
                    for variable_id in tree_node._standard_variable_ids:
                        if tree_node.is_variable_boolean(variable_id):  # for now...
                            diff = scen._x[tree_node._name][variable_id] - xbars[variable_id]
                            sest = (maxs[variable_id] - mins[variable_id]) / 4.0 # close enough to stdev
                            if sest > ph._integer_tolerance: # actually, it should be at least 1/4 if not zero
                                dist = min(3, abs(diff)/sest) # truncated z score
                                if dist > biggestdist:
                                    biggestdist = dist
                                    winner = (tree_node._name, variable_id)
            if biggestdist > 0:
                return biggestdist, stagenum, winner
        return None, None, None

#=========================
    def Select_var_by_latest_convergence(self, ph):
        # assuming the solve is complete, find a good variable on which to branch
        # we want one that converged last 
        # return a boolean to indicate if Conv iters is provided
        LeastConvergedIters = ph._current_iteration+1  # how long have we been converged
        winner = None
        stagenum = 0
        for stage in ph._scenario_tree._stages[:-1]:
            stagenum += 1
            for tree_node in stage._tree_nodes:
                if not hasattr(tree_node, '_num_iters_converged'):
                    print ("warning: _num_iters_converged not provided by wwextensions")
                    return None, (None, None), False
                ConIters = tree_node._num_iters_converged
                for scen in tree_node._scenarios:
                    for variable_id in tree_node._standard_variable_ids:
                        if tree_node.is_variable_boolean(variable_id):  # for now...
                            if not hasattr(tree_node, "_variable_bounds"):
                                self._collect_node_variable_bounds(ph, tree_node)
                            (lbval, ubval) = tree_node._variable_bounds[variable_id]
                            if (lbval is None or ubval is None or lbval != ubval): # don't want a do-over
                                if ConIters[variable_id] < LeastConvergedIters:
                                    LeastConvergedIters = ConIters[variable_id]
                                    winner = (tree_node._name, variable_id)
            if winner is not None:
                ### if options.BBPH_Verbose is True:
                print ("LeastConvergedIters=",LeastConvergedIters)
                return stagenum, winner, True
        return None, (None, None), True

#=========================
    def Select_var_by_first_unbranched(self, ph):
        # assuming the solve is complete, find a variable on which to branch
        # here we also will branch on derived variables

        def Process_var(stage, tree_node, variable_id):
            # return True if this is the var to use
            varname, varindex = tree_node._variable_ids[variable_id]
            #print ("looking at stagenum=%d varid=%s; %s[%s]" % (stagenum, str(variable_id), varname, varindex))
            if tree_node.is_variable_boolean(variable_id):  # as of Aug 2015, TBD: general ints
                ### if options.BBPH_Verbose is True:
                print ("looking hard at stage=%s varid=%s; %s[%s]" % (stage.name, str(variable_id), varname, varindex))
                # note: we need to know if we have branched on it to get here
                if not hasattr(tree_node, "_variable_bounds"):
                    self._collect_node_variable_bounds(ph, tree_node)
                (lbval, ubval) = tree_node._variable_bounds[variable_id]
                return lbval is None or ubval is None or lbval != ubval

        stagenum = 0
        for stage in ph._scenario_tree._stages[:-1]:
            stagenum += 1
            for tree_node in stage._tree_nodes:
                for variable_id in tree_node._standard_variable_ids:
                    is_it = Process_var(stage, tree_node, variable_id)
                    if is_it:
                        return stagenum, tree_node._name, variable_id
                for variable_id in tree_node._derived_variable_ids:
                    is_it = Process_var(stage, tree_node, variable_id)
                    if is_it:
                        return stagenum, tree_node._name, variable_id

        return None, None, None

#==================

    #
    # Store the variable bounds on the tree node by variable id
    # If this is parallel ph, then we need to do this by transmitting
    # an external function invocation that collects the variable
    # bounds from a given list of node names. We do this in such
    # a way as to minimize the amount of data we need to transfer
    # (e.g., in the case of a two stage problem, we request information
    # about the root node from only one of the PH solver servers)
    #

    def _collect_node_variable_bounds(self, ph, tree_node):

        if not isinstance(ph._solver_manager,
                          pyomo.solvers.plugins.\
                          smanager.phpyro.SolverManager_PHPyro):

            tree_node._variable_bounds = \
                collect_node_variable_bounds(tree_node)

        else:
            # NOTE: as of 24 May 2016 this does not work in archiveBBPH either
            # edited by dlw 24 May 2016
            object_name = ph._scenario_tree._scenarios[0]._name
            ah = \
                phsolverserverutils.transmit_external_function_invocation_to_worker(
                    ph,
                    object_name,
                    thisfile,
                    "external_collect_variable_bounds",
                    return_action_handle=True,
                    function_args=((tree_node.name,),))

            results = ph._solver_manager.wait_for(ah)
            # dlw ?? object_name = action_handle_map[ah]
            assert tree_node.name in results
            tree_node._variable_bounds = results[tree_node.name]


