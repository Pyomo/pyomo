# This is research code that can be used to re-create the tests
# reported in the BBPH paper in OR Letters.
#
### DO NOT EDIT ###  !!!!!  !!!!                  !!!!!!  !!!!
# THIS IS FROZEN BY ORDER OF DLW MAY 21, 2016
#
# If we want to branch on non-binary vars, we need var bounds info to detect terminal-ness:
#   but in general, that can vary by scenario, so we would need the min var lb and max var ub
#   but... for now we could just branch and then hit infeasiblity...

import pyomo.environ
from pyomo.pysp import ph, phinit, phutils
from pyomo.core.base import maximize, minimize
import pyomo.solvers.plugins.smanager.phpyro
from pyomo.pysp import phsolverserverutils

import os
from optparse import OptionParser, OptionGroup
import copy
import sys
import time

thisfile = os.path.abspath(__file__)

# branch and bound based on PH

###### local options ######
def AddBBOptions(parser):
    # add BB options to the parser object
    BBPH_options = parser.add_argument_group("BB PH options")

    BBPH_options.add_argument('--BBPH-OuterIterationLimit',
        help='Specifies the maximum number of Outer Iterations. Default is None.',
        action='store',
        dest='BBPH_OuterIterationLimit',
        type=int,
        default=None)

    BBPH_options.add_argument('--BBPH-Verbose',
        help='Specifies verbose output from BBPH. Default is False.',
        action='store_true', 
        dest='BBPH_Verbose',
        default=False)

    BBPH_options.add_argument('--BBPH-Initial-IB',
        help='Specifies an initial value of the global inner bound. Default is None.',
        action='store',
        dest='BBPH_Initial_IB',
        type=float,
        default=None)

    BBPH_options.add_argument('--BBPH-Initial-OB',
        help='Specifies an initial value of the global outer bound. Default is None.',
        action='store',
        dest='BBPH_Initial_OB',
        type=float,
        default=None)

    BBPH_options.add_argument('--BBPH-No-Branch-Epsilon',
        help="For the node IB and OB, an abs(IB-OB) that precludes branching; default 1e-9",
        action='store',
        dest='BBPH_No_Branch_Epsilon',
        type=float,
        default=1e-9)

    BBPH_options.add_argument('--BBPH-Terminate-Epsilon',
        help="For global IB and OB, an abs(IB-OB) that causes termination; default 1e-5",
        action='store',
        dest='BBPH_Terminate_Epsilon',
        type=float,
        default=1e-5)

######### Global Counts dictionary for reporting #######

Counts = {}
Counts["Infeasible"] = 0
Counts["Pruned by Bound Threshhold"] = 0
Counts["Pruned by IB == OB"] = 0        

###### for transmitting variable bounds ####
def external_branch_on_variable(ph, scenario_tree, scenario_or_bundle, boundslist, Verbose):
    # signature needed to pass across to solvers
    if Verbose:
        print ("creating bounds for B&B node for",scenario_or_bundle.name,"as follows:")
    instance = ph._instances[scenario_or_bundle.name]
    for boundtuple in boundslist:
        ((scenario_tree_node_name, variable_id), value, direction) = boundtuple 
        tree_node = scenario_tree.get_node(scenario_tree_node_name)
        varname, varindex = tree_node._variable_ids[variable_id]
        vardataobject = getattr(instance, varname)[varindex]
        if direction == 'up':
            vardataobject.setlb(value)
        else:
            vardataobject.setub(value)
        if Verbose:
            print ("   ",varname,"[",varindex,"] ",direction,value)
    instance.preprocess()

#
# Create a dictionary mapping scenario tree id to
# variable bounds
#
def collect_node_variable_bounds(tree_node):
    assert tree_node._variable_datas is not None
    var_bounds = {}
    for variable_id, vardatas in iteritems(tree_node._variable_datas):
        vardata = vardatas[0][0]
        if not vardata.is_expression():
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

###### output utilities #########
def IBName(sense):
    if sense == minimize:
        return "upper bound "
    elif sense == maximize:
       return "lower bound"
    else:
       return "N/A"

def OBName(sense):
    if sense == maximize:
        return "upper bound"
    elif sense == minimize:
       return "lower bound"
    else:
       return "N/A"

##### comparison utility #####
def isBetter(val0, sense, val1):
   # return true if val0 is better than val1 using sense
   return (sense is minimize and val0 < val1) or (sense is maximize and val0 > val1)

########################### node class ##############################
class BBPH_node(object):
    # assumed to be a node in a binary tree in some places

    def __init__(self, options, BranchTupleList=None, Parent=None, isTerminal=False):
        # only the root BB node has no Parent and no BranchTuple
        # NOTE: if you want to know what branch made this, look at BranchTupleList[-1]

        self._options = options
        self.Parent = Parent
        # the ChildrenOB list is created when we do the branching (which is done all at once)
        # if a child has a "none" bound, it has not been processed yet
        # NOTE: there might be 0,1, or two children
        self.ChildrenOBs = {} # Each child can enter the list and/or update outer bounds
        self.BranchTupleList = BranchTupleList # ((scenario_tree_node_name, variable_id), value, direction)
        self.InnerBound = None
        self.OuterBound = None

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

            new_action_handle = \
                transmit_external_function_invocation_to_worker(
                    ph,
                    object_name,
                    thisfile,
                    "external_collect_variable_bounds",
                    return_action_handle=True,
                    function_args=((tree_node.name,),))

            results = ph._solver_manager.wait_for(ah)
            object_name = action_handle_map[ah]
            assert tree_node.name in results
            tree_node._variable_bounds = results[tree_node.name]

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
                if options.BBPH_Verbose is True:
                    print ("LeastConvergedIters=",LeastConvergedIters)
                return stagenum, winner, True
        return None, (None, None), True

    def Select_var_by_first_unbranched(self, ph):
        # assuming the solve is complete, find a variable on which to branch
        # here we also will branch on derived variables

        def Process_var(stage, tree_node, variable_id):
            # return True if this is the var to use
            varname, varindex = tree_node._variable_ids[variable_id]
            #print ("looking at stagenum=%d varid=%s; %s[%s]" % (stagenum, str(variable_id), varname, varindex))
            notit = True
            if tree_node.is_variable_boolean(variable_id):  # as of Aug 2015, TBD: general ints
                if options.BBPH_Verbose is True:
                    print ("looking hard at stage=%s varid=%s; %s[%s]" % (stage.name, str(variable_id), varname, varindex))
                notit = False
                # note: we need to know if we have branched on it to get here
                if self.BranchTupleList is not None:
                    for BT in self.BranchTupleList:
                        ((tn, vi), v, d) = BT
                        if (tree_node._name, variable_id) == (tn, vi):
                            notit = True
                if not notit:
                    return True
            return False

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

    def ProcessNewChildOB(self, ChildNode, sense):
        # One of my children has a new OB, see if I care
        # I might want to update the kids OB locally, and maybe update my OB
        # If I update my OB, call this function on my parent (if I have one)
        
        currkidval = self.ChildrenOBs[ChildNode]
        if currkidval is not None \
           and not isBetter(ChildNode.OuterBound, sense, currkidval):
           return  # we are not moved to act on this

        if options.BBPH_Verbose is True:
            print ("Updating OB for child in node")
        self.ChildrenOBs[ChildNode] = ChildNode.OuterBound

        ValtoUse = self.OuterBound
        # note: this could be the final (i.e. second) arrival
        for kid in self.ChildrenOBs: # for sort of would work with non-binary (but full-partition) trees...
            if self.ChildrenOBs[kid] is None:
                if options.BBPH_Verbose is True:
                    print ("Not all children have reported OB so no need to update further")
                return
            if isBetter(self.ChildrenOBs[kid], sense, ValtoUse):
                ValtoUse = self.ChildrenOBs[kid]

        if isBetter(ValtoUse, sense, self.OuterBound):
            if options.BBPH_Verbose is True:
                print ("Updating OB for this node")
            self.OuterBound = ChildNode.OuterBound
        else:
            if options.BBPH_Verbose is True:
                ("No OB improment for this node")
            return

        if self.Parent is not None:
            if options.BBPH_Verbose is True:
                print ("Passing my new OB to my parent")
            self.Parent.ProcessNewChildOB(self, sense)
        
    def Branch(self, ph):
        # pick a variable and return two new BBPH_nodes
        # i.e. do all binary branching on the node

        # see if ph reached an integer xhat in the first stage
        # not terribly efficient...
        dist, stagenum, varpair = self.Select_var_by_dist_from_xbar(ph)
        if stagenum is None:
            if options.BBPH_Verbose is True:
                print ("BBPH: All at xbar, so trying latest converged")
            latestSN, (latestTN, latestVI), couldlatest = self.Select_var_by_latest_convergence(ph)
            if not couldlatest:
                if options.BBPH_Verbose is True:
                    print ("resorting to unbranched")
                # use the vars with "latest" names even though that is not the correct name...
                latestSN, latestTN, latestVI = self.Select_var_by_first_unbranched(ph)
            if latestSN is None:  # we struck out...
                return None, None
        else:
            latestSN = None  # good housekeeping
        if stagenum == None or (stagenum != None and latestSN != None and latestSN < stagenum):
            # we want the latest or the unbranched because they are all we have or an earlier stage
            if options.BBPH_Verbose is True:
                print ("Based on latest converged or unbranched")
            varpair = (latestTN, latestVI)
        else:
            if options.BBPH_Verbose is True:
                print ("Based on dist from xbar")
        node_name, variable_id = varpair
        tree_node = ph._scenario_tree._tree_node_map[node_name]
        varname, varindex = tree_node._variable_ids[variable_id]
        if options.BBPH_Verbose is True:
            print ("BBPH is branching on node=%s varid=%s; %s[%s]" % (node_name, str(variable_id), varname, varindex))
        #xbars = tree_node._xbars
        #xbar = xbars[variable_id]
        # major hack: just do binaries
        BPL = copy.deepcopy(self.BranchTupleList)
        if options.BBPH_Verbose is True:
            print ("Creating branch ", 0, 'down')
        BPL.append(((node_name, variable_id), 0, 'down'))
        bbnodedown = BBPH_node(options, BranchTupleList=BPL, Parent=self, isTerminal=False)
        self.ChildrenOBs[bbnodedown] = None
        BPL = copy.deepcopy(self.BranchTupleList)
        if options.BBPH_Verbose is True:
            print ("Creating branch ", 1, 'up')
        BPL.append(((node_name, variable_id), 1, 'up'))
        bbnodeup = BBPH_node(options, Parent=self, BranchTupleList=BPL, isTerminal=False)
        self.ChildrenOBs[bbnodeup] = None

        return bbnodeup, bbnodedown        

    def Process_Node(self, GIB, NoBranchEpsilon):
        # GIB is the global inner bound (upper for min)
        # don't bother to branch if OB and GIB are within NoBranchEpsilon

        # create a ph object and run the solver; 
        # return five values, but some or all might be None
        # if feasible and not fathomed
        #     return terminalflag, Outerbound, InnerBound, xhat
        #     if not terminal and abs(inner_bound - outer_bound) > tolerance:
        #         return two new BB_nodes
        # Note: you could look at the node and see if it is terminal before processing

        # create the ph object
        # remember to terminate on LB above global UB (or the opposite)
        if GIB != None:
            if options.BBPH_Verbose is True:
                print ("Setting PH outer-bound-convergence=",GIB," which is the global inner bound")
            options.enable_outer_bound_convergence = True
            options.outer_bound_convergence_threshold = GIB

        with phinit.PHFromScratchManagedContext(self._options) as phobject:

            # because the plugins "live" across PH instances, we need
            # to reset them after creating a new PH object. otherwise,
            # they will inherit prior PH execution state.
            phutils.reset_ph_plugins(phobject)

            sense = phobject._objective_sense # sort of wierd that we have to pass this out...

            # TBD: capture the current bounds for the vars in question

            # transmit the var bounds
            if self.BranchTupleList is not None:
                if isinstance(phobject._solver_manager,
                              pyomo.solvers.plugins.smanager.\
                              phpyro.SolverManager_PHPyro):
                    ahs = phsolverserverutils.transmit_external_function_invocation(
                        phobject,
                        thisfile,
                        "external_branch_on_variable",
                        invocation_type=(phsolverserverutils.InvocationType.\
                                         PerScenarioInvocation),
                        return_action_handles=True,
                        function_args=(self.BranchTupleList, options.BBPH_Verbose,))
                    phobject._solver_manager.wait_all(ahs)
                else:
                    for scenario in phobject._scenario_tree.scenarios:
                        external_branch_on_variable(phobject,
                                                    phobject._scenario_tree,
                                                    scenario,
                                                    self.BranchTupleList,
                                                    options.BBPH_Verbose )

            # solve and process solution
            phretval = phobject.solve()

            # TBD: reset var the bounds (see tbd for capture)

            if phretval is not None:
                Counts["Infeasible"] += 1
                if options.BBPH_Verbose is True:
                    print("PH Iteration zero solve was not successful for scenario: "+str(phretval))
                return sense, None, None, None, None, None

            self.InnerBound = IB = phobject._best_reported_inner_bound
            self.OuterBound = OB = phobject._best_reported_outer_bound
            Xhat = copy.deepcopy(phobject._xhat)

            if self.Parent is not None and OB is not None:
                if options.BBPH_Verbose is True:
                    print ("Sending OB to parent node for processing")
                self.Parent.ProcessNewChildOB(self, sense)

            # don't branch if IB==OB or if it terminated on bound threshold
            if OB is not None and GIB is not None and \
                ((sense == minimize and OB > GIB) or (sense == maximize) and OB < GIB):
                Counts["Pruned by Bound Threshhold"] += 1
                if options.BBPH_Verbose is True:
                    print ("Not branching; pruned by global ",IBName(sense))
                return sense, OB, IB, None, None, None # pruned

            if IB is not None and OB is not None and abs(IB-OB) <= NoBranchEpsilon:
                Counts["Pruned by IB == OB"] += 1
                if options.BBPH_Verbose is True:
                    print ("not branching because IB=%f, OB=%f and NoBranchEpsilon = %f"
                            % (IB, OB, NoBranchEpsilon))
                return sense, OB, IB, Xhat, None, None # terminal node

            # branch
            bbnodeup, bbnodedown = self.Branch(phobject)

            if options.BBPH_Verbose is True:
                print ("Returning from Process_node and reporting OB,IB=",OB,IB)
            return sense, OB, IB, Xhat, bbnodedown, bbnodeup


########### BB Node List Processing (e.g., searching for interesting nodes) ######
def Furthest_Outer_Bound(ndList, sense):
    Furthest = None
    for nd in ndList:
        if nd.OuterBound is not None and (Furthest is None or isBetter(Furthest.OuterBound, sense, ndList.OuterBound)):
            Furthest = nd

def Most_Promising_Node(ndList, GIB, GOB, sense):
    # search the node list using global inner bound GIB and outer bound GOB
    return Furthest_Outer_Bound(ndList, sense) # for now...

if __name__ == "__main__":

    ##======================= Main ========================
    # two lists: active nodes and processed nodes
    # when a list is taken off the active list, it is processed, added to the processed list
    # and deleted from the active list. If it spawns children, they are added to the active list
    # NOTE: really, there are two active lists: one "normal" and one for so-called terminal nodes

    start_time = time.time()

    try:
        ph_options_parser = phinit.construct_ph_options_parser("python ./BBPH.py [options]")
        AddBBOptions(ph_options_parser)
        ###(options, args) = ph_options_parser.parse_args(args=sys.argv)
        options = ph_options_parser.parse_args(args=sys.argv[1:])
    except SystemExit as _exc:
        #? the parser throws a system exit if "-h" is specified - catch
        #? it to exit somewhat gracefully.
        sys.exit(0)
        #pass
        ###return _exc.code

    if options.enable_outer_bound_convergence == True:
        print ("\nWARNING: the outer-bound-convergence option will be overwritten\n")

    ActiveNodeList = []
    TerminalNodeList = []
    ProcessedNodeList = []
    GlobalInnerBound = options.BBPH_Initial_IB # None # best
    GlobalXhat = None # best
    GlobalOuterBound = options.BBPH_Initial_OB # None # furthest
    emptyBPL = []
    NoBranchEpsilon = options.BBPH_No_Branch_Epsilon  # an absolute abs(IB-OB)

    def Gap_met():
        # return True (and talk about it) if the global bound difference is inside the gap
        if GlobalInnerBound is not None and GlobalOuterBound is not None \
            and abs(GlobalInnerBound - GlobalOuterBound) <= options.BBPH_Terminate_Epsilon:
            print ("Terminating because GlobalIB=%f, GlobalOB=%f and Terminate-Epsilon = %f"
                        % (GlobalIB, GlobalOB, options.BBPH_Terminate_Epsilon))
            return True
        return False

    RootNode = BBPH_node(options,emptyBPL) # create the rootnode
    ActiveNodeList.append(RootNode)

    OuterIter = 0  # simple iteration count
    IterLim = options.BBPH_OuterIterationLimit

    while len(ActiveNodeList) > 0 and (IterLim is None or OuterIter <= IterLim) and not Gap_met():
        OuterIter += 1
        print ("BBPH OuterIter=",OuterIter)
        bbnode = ActiveNodeList[0]
        sense, OB, IB, Xhat, nd0, nd1 = bbnode.Process_Node(GlobalInnerBound, NoBranchEpsilon)
        if IB is not None and (GlobalInnerBound is None or isBetter(IB, sense, GlobalInnerBound)):
            GlobalInnerBound = IB
            print ("New best global (",IBName(sense),") GlobalInnerBound=",GlobalInnerBound)
            GlobalXhat = Xhat
        if OB is not None:
            # propogate OB info up... TBD
            # If the new OB bound is a "worse" value (tighter; closer to IB; better as a bound)
            if GlobalOuterBound is None or isBetter(GlobalOuterBound, sense, RootNode.OuterBound):
                GlobalOuterBound = RootNode.OuterBound
                print ("New global (",OBName(sense),") GlobalOuterBound=",GlobalOuterBound)
        if Xhat is not None and nd0 is None and nd1 is None:
            TerminalNodeList.append(bbnode)
        else:
            ProcessedNodeList.append(bbnode)
        if nd0 is not None:
            ActiveNodeList.append(nd0)
        if nd1 is not None:
            ActiveNodeList.append(nd1)
        del ActiveNodeList[0]
        print ("The active node list has %d entries" % (len(ActiveNodeList)))
        print ("There are now %d terminal nodes." % (len(TerminalNodeList)))
        print ("Time so far is %f seconds." % (time.time() - start_time))

    end_time = time.time()
    print ("Final Global",IBName(sense)," ,", str(GlobalInnerBound))
    print ("Final Global",OBName(sense)," ,", str(GlobalOuterBound))

    print ("Begin Final Summary.")
    print ("Final Obj,", str(GlobalInnerBound))
    print ("Final Bound,", str(GlobalOuterBound))
    print ("Final elapsed time (sec), %f" % (end_time - start_time))
    print ("Final Outer Iters,", str(OuterIter))
    print ("Final Terminal Node Count,",str(len(TerminalNodeList)))
    print ("Final Active Node Count,",str(len(ActiveNodeList)))
    for c in Counts:
        print (c,",",Counts[c])
    print ("Input IterLim,",str(IterLim))
    print ("Input Termination Epsilon,",str(options.BBPH_Terminate_Epsilon))
    print ("End Final Summary.")

    TLlen = len(TerminalNodeList)
    print ("There are %d terminal nodes to process." % TLlen)

    # we are going to solve the EF for the terminal nodes (note that leaf integers will not be fixed)
    #NOTE: you don't need to re-process nodes with IB and OB close enough together
    if TLlen > 0:
        print ("terminal nodes")
        print ("Index: ", IBName(sense), OBName(sense))
        for ndnum in range(TLlen): # want index number
            nd = TerminalNodeList[ndnum]
            print (ndnum, ": ", nd.InnerBound," , ",nd.OuterBound)
        print ("Branching lists for terminal nodes")
        for ndnum in range(TLlen):
            nd = TerminalNodeList[ndnum]
            print (ndnum,":")
            for bt in nd.BranchTupleList:
                print (bt)
            print("----")
