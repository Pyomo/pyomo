#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

#
# Utility classes for dealing with the COIN-OR optimization
# services
#

import os
import sys

from pyomo.misc.plugin import alias
import pyutilib.math

from pyomo.opt.base import results
from pyomo.opt.base.formats import *
from pyomo.opt import SolverResults, SolutionStatus, SolverStatus, TerminationCondition, MapContainer
from pyomo.os.OSrL import OSrL

def xml_eval(vstring):
    vstring = vstring.strip()
    if vstring == 'NaN':
        return pyutilib.math.nan
    elif vstring == 'zero':
        return 0
    elif vstring == 'INF':
        return pyutilib.math.infinity
    return eval(vstring)

class ResultsReader_osrl(results.AbstractResultsReader):
    """
    Class that reads in an OSrL results file and generates a
    SolverResults object.
    """

    alias(str(ResultsFormat.osrl))

    def __init__(self, name=None):
        results.AbstractResultsReader.__init__(self,ResultsFormat.osrl)
        if not name is None:                            #pragma:nocover
            self.name = name

    def __call__(self, filename, res=None, suffixes=None):
        if res is None:
            res = SolverResults()
        osrl = OSrL()
        osrl.read(filename)
        #
        # Process 'general' element
        #
        doc = osrl.etree.getroot()
        e = doc.find('.//{%s}general' % osrl.namespace)
        if not e is None:
            for elt in ['message', 'serviceURI', 'serviceName', 'jobID', 'solverInvoked', 'timeStamp']:
                value = e.findtext('.//{%s}%s' % (osrl.namespace,elt))
                if not value is None:
                    setattr(res.solver, elt, value)
            status = e.find('.//{%s}generalStatus' % osrl.namespace).attrib['type']
            if status == 'normal':
                res.solver.status = SolverStatus.ok
            else: 
                try:
                    res.solver.status = SolverStatus(status)
                except:
                    res.solver.status = SolverStatus.error
            res.problem.name = e.findtext('.//{%s}instanceName' % osrl.namespace)
        #
        # Process 'job' element
        #
        e = osrl.etree.find('.//{%s}job' % osrl.namespace)
        if not e is None:
            res.add('job', MapContainer(), False, "Job Execution")
            for elt in ['status', 'submitTime', 'scheduledStartTime', 'actualStartTime', 'endTime', 'usedCPUSpeed', 'usedCPUNumber', 'usedDiskSpace', 'usedMemory']:
                value = e.findtext('.//{%s}%s' % (osrl.namespace,elt))
                if not value is None:
                    setattr(res.solver, elt, value)
            e = e.find('.//{%s}time' % osrl.namespace)
            if not e is None:
                setattr(res.job, e.attrib['type'], xml_eval(e.text))
        #
        # Process 'optimization' element
        #
        e = osrl.etree.find('.//{%s}optimization' % osrl.namespace)
        if not e is None:
            res.problem.number_of_constraints = xml_eval(e.attrib['numberOfConstraints'])
            res.problem.number_of_variables = xml_eval(e.attrib['numberOfVariables'])
            res.problem.number_of_objectives = xml_eval(e.attrib['numberOfObjectives'])
            for soln in e.findall('.//{%s}solution' % osrl.namespace):
                #
                # We create a solution, and then populate its data
                # The solution object is managed by the SolverResults instance
                #
                solution = res.solution.add()
                solution.status = SolutionStatus(soln.find('.//{%s}status' % osrl.namespace).attrib['type'])
                solution.message = soln.findtext('.//{%s}message' % osrl.namespace)
                #
                vars = soln.find('.//{%s}variables' % osrl.namespace)
                if not vars is None:
                    values = vars.find('.//{%s}values' % osrl.namespace)
                    if not values is None:
                        for var in values.findall('.//{%s}var' % osrl.namespace):
                            id = int(var.attrib['idx'])
                            solution.variable['x'+str(id)] = {"Value" : xml_eval(var.text), "Id" : id}
                    #
                    values = vars.find('.//{%s}valueString' % osrl.namespace)
                    if not values is None:
                        for var in values.findall('.//{%s}var' % osrl.namespace):
                            id = int(var.attrib['idx'])
                            solution.variable['x'+str(id)] = {"Value" : xml_eval(var.text), "Id" : id}
                    #
                    for other in vars.findall('.//{%s}other' % osrl.namespace):
                        if other.attrib['name'] == 'name':                  #pragma:nocover
                            continue
                        for var in other.findall('.//{%s}var' % osrl.namespace):
                            id = int(var.attrib['idx'])
                            if not 'x'+str(id) in solution.variable:
                                solution.variable['x'+str(id)] = {"Id" : id}
                            solution.variable['x'+str(id)][other.attrib['name']] = xml_eval(var.text)
                #
                cons = soln.find('.//{%s}constraints' % osrl.namespace)
                duals = cons.find('.//{%s}dualValues' % osrl.namespace)
                if not duals is None:
                    for dual in duals.findall('.//{%s}con' % osrl.namespace):
                        id = int(dual.attrib['idx'])
                        solution.constraint['x'+str(id)] = {"Dual" : xml_eval(dual.text), "Id" : id}
                #
                objs = soln.find('.//{%s}objectives' % osrl.namespace)
                if not objs is None:
                    values = objs.find('.//{%s}values' % osrl.namespace)
                    for obj in values.findall('.//{%s}obj' % osrl.namespace):
                        try:
                            id = -int(obj.attrib['idx'])-1
                        except:
                            id = 0
                        solution.objective['x'+str(id)] = xml_eval(obj.text)
                    #
                    for other in objs.findall('.//{%s}other' % osrl.namespace):
                        if other.attrib['name'] == 'name':              #pragma:nocover
                            continue
                        for obj in other.findall('.//{%s}obj' % osrl.namespace):
                            try:
                                id = -int(obj.attrib['idx'])-1
                            except:
                                id = 0
                            if not 'x'+str(id) in solution.objective:
                                solution.objective['x'+str(id)].value = 0
                            solution.objective['x'+str(id)][other.attrib['name']] = xml_eval(obj.text)
        #
        return res
