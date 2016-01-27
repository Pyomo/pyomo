"""
produce a report of variables sorted by "bad W behavior"
"""
#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# sorg W: a plugin that cares about the W vectors
#

import sys

import pyomo.util.plugin
from pyomo.core import *
from pyomo.pysp import phextension
from pyomo.pysp.phutils import *
from pyomo.pysp.generators import \
    scenario_tree_node_variables_generator_noinstances

#==================================================
class sorgwextension(pyomo.util.plugin.SingletonPlugin):

    pyomo.util.plugin.implements(phextension.IPHExtension)
    
    # the below is a hack to get this extension into the
    # set of IPHExtension objects, so it can be queried
    # automagically by PH.
    pyomo.util.plugin.alias("sorgwextension")

    def __init__(self, *args, **kwds):

        self.Tol = 1e-6
        self.wtrace_filename = 'sorgw.ssv'
        self.wsummary_filename = 'wsummary.ssv'
        self.winterest_filename = 'winterest.ssv' # only vars of interest
        ####### Thresholds for interestingness  
        ####### (ored, so any zero causes all to be interesting)
        self.threshWZeroCrossings = 2
        self.threshDiffsRatio = 0.2
        self.threshDiffZeroCrossings = 3

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
        
        print("sorgw.py is writing the semi-colon separated values file "+self.wtrace_filename)
        ofile = open(self.wtrace_filename, "w")
        self._w_printing(ofile) # write the header
        self._w_printing(ofile, ph)
        ofile.close()

#==================================================

    def pre_iteration_k_solves(self, ph):
        pass

#==================================================
    def post_iteration_k_solves(self, ph):
        pass

#==================================================
    def post_iteration_k(self, ph):

        ofile = open(self.wtrace_filename, "a")
        self._w_printing(ofile, ph)
        ofile.close()

#==================================================
    def post_ph_execution(self, ph):
        # note that we could keep it all in memory and not use a file
        W_Traces = self.Read_W_Traces(self.wtrace_filename)
        self.Compute_and_Write_it_all(W_Traces)

#=========================
    def _w_printing(self, ofile, ph=None):
        # print the w values in a useful way to the open file ofile
        # if ph is None, just write the header
        if ph is None:
            ofile.write("iteration; tree node; scenario; variable; W\n")
        else:
            for stage, tree_node, variable_id, variable_values, is_fixed, is_stale in \
                scenario_tree_node_variables_generator_noinstances(ph._scenario_tree,
                                                                   includeDerivedVariables=False,
                                                                   includeLastStage=False):
                if (is_stale is False):
                    for scenario in tree_node._scenarios:
                       scen_name = scenario._name
                       weight_value = scenario._w[tree_node._name][variable_id]
                       variable_name, index = tree_node._variable_ids[variable_id]
                       full_variable_name = variable_name+indexToString(index)
                       ofile.write(str(ph._current_iteration) + ';' + tree_node._name + ';' + scen_name + ';' + full_variable_name + ';' + str(weight_value)+'\n')


    #########
    def Read_W_Traces(self, fname):
        # read the W_Traces as written by sorgw.py
        # don't really check for input errors other than a bad iter order

        def num(s):
            try:
                return int(s)
            except ValueError:
                return float(s)

        infile = open(fname,"r")
        curriter = 0
        W_Traces = {}
        for linein in infile:
            parts = linein.split(';')
            if parts[0] == 'iteration': # hack to skip header
                continue
            iternum = int(parts[0])
            scenname = parts[2]
            varname = parts[3]
            wval = num(parts[4])
            if iternum != curriter:
                if iternum-curriter == 1:
                    curriter = iternum
                else:
                    print ("HEY! the input in "+fname+" has iter "+str(iternum)+" after "+str(curriter)+'\n')
                    print (linein)
                    sys.exit(1)
            if varname not in W_Traces:
                W_Traces[varname] = {}
            if scenname not in W_Traces[varname]:
                W_Traces[varname][scenname] = []
            W_Traces[varname][scenname].append(wval)
        return W_Traces

    #####
    def Score_a_Trace(self, wtrace):
        # given a list of w values, compute and return scores
        # (lower is better)
        def dlwmean(v):
            if len(v) > 0:
                return sum(v)/len(v)
            else:
                return 0
        WZeroCrossings = 0  # a score
        WAbove = False
        WBelow = False
        # note: hitting zero does not reset Above or Below or cause a crossing
        for w in wtrace:
            if abs(w) < self.Tol:
                continue
            if w > self.Tol:
                if WBelow:
                    WZeroCrossings += 1
                    WBelow = False
                WAbove = True
            if w < self.Tol:
                if WAbove:
                    WZeroCrossings += 1
                    WAbove = False
                WBelow = True

        DiffsRatio = 0  # a score
        DiffZeroCrossings = 0  # a score
        absdiffs = [] # t to t+1
        DiffAbove = False
        DiffBelow = False
        wcnt = len(wtrace)
        for i in range(wcnt-1):
            diff = wtrace[i+1] - wtrace[i]
            absdiffs.append(abs(diff))
            if abs(diff) < self.Tol:
                continue
            if diff > self.Tol:
                if DiffBelow:
                    DiffZeroCrossings += 1
                    DiffBelow = False
                DiffAbove = True
            if diff < self.Tol:
                if DiffAbove:
                    DiffZeroCrossings += 1
                    DiffAbove = False
                DiffBelow = True
        frontavg = dlwmean(absdiffs[:int(wcnt/2)])
        if frontavg > self.Tol:
            DiffsRatio = dlwmean(absdiffs[int(wcnt/2):]) / frontavg # close enough
        return WZeroCrossings, DiffsRatio, DiffZeroCrossings

    ####
    def Of_Interest(self, WZeroCrossings, DiffsRatio, DiffZeroCrossings):
    # return true if anything is above its threshold
        return WZeroCrossings >= self.threshWZeroCrossings or \
               DiffsRatio >= self.threshDiffsRatio or \
               DiffZeroCrossings >= self.threshDiffZeroCrossings

    ####
    def Compute_and_Write_it_all(self, W_Traces):
        VarsOfInterest = set()
        fname = self.wsummary_filename
        print ("sorgw.py is writing the semi-colon separated values file "+fname)
        ofile = open(fname, "w")
        ofile.write("var; scen; WZeroCrossing; DiffsRatio; DiffZeroCrossings; w values...\n")
        for varname in W_Traces:
            for scenname in W_Traces[varname]:
                WZeroCrossings, DiffsRatio, DiffZeroCrossings = self.Score_a_Trace(W_Traces[varname][scenname])
                if self.Of_Interest(WZeroCrossings, DiffsRatio, DiffZeroCrossings):
                    VarsOfInterest.add(varname)
                ofile.write(varname+';'+scenname+';'+str(WZeroCrossings)+';'+str(DiffsRatio)+';'+str(DiffZeroCrossings))
                for w in W_Traces[varname][scenname]:
                    ofile.write(';'+str(w))
                ofile.write('\n')
        ofile.close
        fname = self.winterest_filename
        print ("sorgw.py is writing the semi-colon separated values file "+fname)
        ofile = open(fname, "w")
        ofile.write("var; scen; WZeroCrossing; DiffsRatio; DiffZeroCrossings; w values...\n")
        for varname in VarsOfInterest:
            for scenname in W_Traces[varname]:
                WZeroCrossings, DiffsRatio, DiffZeroCrossings = self.Score_a_Trace(W_Traces[varname][scenname])
                ofile.write(varname+';'+scenname+';'+str(WZeroCrossings)+';'+str(DiffsRatio)+';'+str(DiffZeroCrossings))
                for w in W_Traces[varname][scenname]:
                    ofile.write(';'+str(w))
                ofile.write('\n')
        ofile.close
