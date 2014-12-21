#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

#
# The formats that are supported by Pyomo
#
__all__ = ['ProblemFormat', 'ResultsFormat', 'guess_format']

from pyutilib.enum import Enum

#
# pyomo - A pyomo.core.PyomoModel object, or a *.py file that defines such an object
# cpxlp - A CPLEX LP file
# nl - AMPL *.nl file
# mps - A free-format MPS file
# mod - AMPL *.mod file
# lpxlp - A LPSolve LP file
# osil - An XML file defined by the COIN-OR OS project: Instance
# FuncDesigner - A FuncDesigner problem
# colin - A COLIN shell command
# colin_optproblem - A Python object that inherits from
#                   pyomo.opt.colin.OptProblem (this can wrap a COLIN shell
#                   command, or provide a runtime optimization problem)
# bar - A Baron input file
#
ProblemFormat = Enum('colin', 'pyomo', 'cpxlp', 'nl', 'mps', 'mod', 'lpxlp', 'osil', 'colin_optproblem', 'FuncDesigner','bar')

#
# osrl - osrl XML file defined by the COIN-OR OS project: Result
# results - A Pyomo results object  (reader define by solver class)
# sol - AMPL *.sol file
# soln - A solver-specific solution file  (reader define by solver class)
# yaml - A Pyomo results file in YAML format
# json - A Pyomo results file in JSON format
#
ResultsFormat = Enum('osrl', 'results', 'sol', 'soln', 'yaml', 'json')


def guess_format(filename):
    formats = {}
    formats['py']=ProblemFormat.pyomo
    formats['nl']=ProblemFormat.nl
    formats['bar']=ProblemFormat.bar
    formats['mps']=ProblemFormat.mps
    formats['mod']=ProblemFormat.mod
    formats['lp']=ProblemFormat.cpxlp
    formats['osil']=ProblemFormat.osil
    formats['sol']=ResultsFormat.sol
    formats['osrl']=ResultsFormat.osrl
    formats['soln']=ResultsFormat.soln
    formats['yml']=ResultsFormat.yaml
    formats['yaml']=ResultsFormat.yaml
    formats['jsn']=ResultsFormat.json
    formats['json']=ResultsFormat.json
    formats['results']=ResultsFormat.yaml
    for fmt in formats:
        if filename.endswith('.'+fmt):
            return formats[fmt]
    return None
