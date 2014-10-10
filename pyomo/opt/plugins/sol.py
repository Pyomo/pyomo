#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

#
# Class for reading an AMPL *.sol file
#

import os
import sys
import re

import pyutilib.misc

from pyomo.misc.plugin import alias
from pyomo.opt.base import results
from pyomo.opt.base.formats import *
from pyomo.opt import SolverResults, SolutionStatus, SolverStatus, TerminationCondition

from six.moves import xrange

class ResultsReader_sol(results.AbstractResultsReader):
    """
    Class that reads in a *.sol results file and generates a
    SolverResults object.
    """

    alias(str(ResultsFormat.sol))

    def __init__(self, name=None):
        results.AbstractResultsReader.__init__(self,ResultsFormat.sol)
        if not name is None:
            self.name = name

    def __call__(self, filename, res=None, soln=None, suffixes=[]):
        """
        Parse a *.sol file
        """

        if res is None:
            res = SolverResults()
        #
        IN = open(filename,"r")
        msg = ""
        line = IN.readline()
        while line:
            if line[0] == '\n' or (line[0] == '\r' and line[1] == '\n'):
                break
            msg += line
            line = IN.readline()
        z = []
        line = IN.readline()
        if line[:7] == "Options":
            line = IN.readline()
            nopts = int(line)
            need_vbtol = False
            if nopts > 4:           # WEH - when is this true?
                nopts -= 2
                need_vbtol = True
            for i in range(nopts + 4):
                line = IN.readline()
                z += [int(line)]
            if need_vbtol:          # WEH - when is this true?
                line = IN.readline()
                z += [float(line)]
        else:
            IN.close()
            msg = "Error reading \"" + filename + "\": no Options line found.\nSOL File Output:\n"
            IN = open(filename, 'r')
            for line in IN:
                msg += line
            IN.close()
            raise ValueError(msg)
        n = z[nopts + 3] # variables
        m = z[nopts + 1] # constraints
        x = []
        y = []
        i = 0
        while i < m:
            line = IN.readline()
            y.append(float(line))
            i += 1
        i = 0
        while i < n:
            line = IN.readline()
            x.append(float(line))
            i += 1
        objno = [0,0]
        line = IN.readline()
        if line:                    # WEH - when is this true?
            if line[:5] != "objno":         #pragma:nocover
                raise ValueError("Error reading \"" + filename + "\": expected \"objno\", found", line)
            t = line.split()
            if len(t) != 3:
                raise ValueError("Error reading \"" + filename + "\": expected two numbers in objno line, but found", line)
            objno = [int(t[1]), int(t[2])]
        res.solver.message = msg.strip()
        res.solver.message = res.solver.message.replace("\n","; ")
        res.solver.message = pyutilib.misc.yaml_fix(res.solver.message)
        ##res.solver.instanceName = osrl.header.instanceName
        ##res.solver.systime = osrl.header.time
        res.solver.status = SolverStatus.ok
        soln_status = SolutionStatus.unknown
        if (objno[1] >= 0) and (objno[1] <= 99):
            objno_message = "OPTIMAL SOLUTION FOUND!"
            res.solver.termination_condition = TerminationCondition.optimal
            res.solver.status = SolverStatus.ok
            soln_status = SolutionStatus.optimal
        elif (objno[1] >= 100) and (objno[1] <= 199):
            objno_message = "Optimal solution indicated, but ERROR LIKELY!"
            res.solver.termination_condition = TerminationCondition.optimal
            res.solver.status = SolverStatus.warning
            soln_status = SolutionStatus.optimal
        elif (objno[1] >= 200) and (objno[1] <= 299):
            objno_message = "INFEASIBLE SOLUTION: constraints cannot be satisfied!"
            res.solver.termination_condition = TerminationCondition.infeasible
            res.solver.status = SolverStatus.warning
            soln_status = SolutionStatus.infeasible
        elif (objno[1] >= 300) and (objno[1] <= 399):
            objno_message = "UNBOUNDED PROBLEM: the objective can be improved without limit!"
            res.solver.termination_condition = TerminationCondition.unbounded
            res.solver.status = SolverStatus.warning
            soln_status = SolutionStatus.unbounded
        elif (objno[1] >= 400) and (objno[1] <= 499):
            objno_message = "EXCEEDED MAXIMUM NUMBER OF ITERATIONS: the solver was stopped by a limit that you set!"
            res.solver.termination_condition = TerminationCondition.maxIterations
            res.solver.status = SolverStatus.warning
            soln_status = SolutionStatus.stoppedByLimit
        elif (objno[1] >= 500) and (objno[1] <= 599):
            objno_message = "FAILURE: the solver stopped by an error condition in the solver routines!"
            res.solver.termination_condition = TerminationCondition.internalSolverError
            res.solver.status = SolverStatus.error
            soln_status = SolutionStatus.error
        res.solver.id = objno[1]
        ##res.problem.name = osrl.header.instanceName
        if res.solver.termination_condition in [TerminationCondition.unknown,
                        TerminationCondition.maxIterations,
                        TerminationCondition.minFunctionValue,
                        TerminationCondition.minStepLength,
                        TerminationCondition.globallyOptimal,
                        TerminationCondition.locallyOptimal,
                        TerminationCondition.optimal,
                        TerminationCondition.maxEvaluations,
                        TerminationCondition.other,
                        TerminationCondition.infeasible]:

            if soln is None:
                soln = res.solution.add()
            res.solution.status = soln_status
            soln.status_description = objno_message
            soln.message = msg.strip()
            soln.message = res.solver.message.replace("\n","; ")
            i = 0
            soln_variable = soln.variable
            for var_value in x:
                soln_variable["v"+str(i)] = {"Value" : var_value, "Id" : i}
                i += 1

            if any(re.match(suf,"dual") for suf in suffixes):
                for i in range(0,len(y)):
                    soln.constraint["c"+str(i)] = {"Dual" : y[i], "Id" : i}

            ### Read suffixes ###
            line = IN.readline()
            line = line.strip()
            while (line and len(line) > 0):
                line = line.split()
                assert line[0] == 'suffix'
                unmasked_kind = int(line[1])
                kind = unmasked_kind & 3 # 0-var, 1-con, 2-obj, 3-prob
                convert_function = int
                if (unmasked_kind & 4) == 4:
                    convert_function = float
                nvalues = int(line[2])
                namelen = int(line[3])
                tablen = int(line[4])
                tabline = int(line[5])
                suffix_name = IN.readline().strip()
                if any(re.match(suf,suffix_name) for suf in suffixes):
                    # ignore translation of the table number to string value for now,
                    # this information can be obtained from the solver documentation
                    for n in xrange(tabline):
                        IN.readline()
                    if kind == 0: # Var
                        for cnt in range(nvalues):
                            suf_line = IN.readline().split()
                            soln_variable.get("v"+suf_line[0],{})[suffix_name] = convert_function(suf_line[1])
                    elif kind == 1: # Con
                        for cnt in range(nvalues):
                            suf_line = IN.readline().split()
                            key = "c"+suf_line[0]
                            if key not in soln.constraint:
                                soln.constraint[key] = {"Id" : len(soln.constraint)}
                            # convert the first letter of the suffix name to upper case,
                            # mainly for pretty-print / output purposes. these are lower-cased
                            # when loaded into real suffixes, so it is largely redundant.
                            translated_suffix_name = suffix_name[0].upper() + suffix_name[1:]
                            soln.constraint[key][translated_suffix_name] = convert_function(suf_line[1])
                    elif kind == 2: # Obj
                        for cnt in range(nvalues):
                            suf_line = IN.readline().split()
                            soln.objective["o"+suf_line[0]][suffix_name] = convert_function(suf_line[1])
                    elif kind == 3: # Prob
                        # Skip problem kind suffixes for now. Not sure the
                        # best place to put them in the results object
                        for cnt in range(nvalues):
                            suf_line = IN.readline().split()
                            soln.problem[suffix_name] = convert_function(suf_line[1])
                else:
                    # do not store the suffix in the solution object
                    for cnt in range(nvalues):
                        IN.readline()
                line = IN.readline()
                line = line.strip()
        ###
        IN.close()
        #
        # This is a bit of a hack to accommodate PICO.  If
        # the PICO parser has parsed the # of constraints, then
        # don't try to read it in from the *.sol file.  The reason
        # is that these may be inconsistent values!
        #
        if res.problem.number_of_constraints == 0:
            res.problem.number_of_constraints = m
        res.problem.number_of_variables = n
        res.problem.number_of_objectives = 1
        return res
