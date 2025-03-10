#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import enum


#
# pyomo - A pyomo.core.PyomoModel object, or a *.py file that defines such an object
# cpxlp - A CPLEX LP file
# nl - AMPL *.nl file
# mps - A free-format MPS file
# mod - AMPL *.mod file
# lpxlp - A LPSolve LP file
# osil - An XML file defined by the COIN-OR OS project: Instance
# bar - A Baron input file
# gams - A GAMS input file
#
class ProblemFormat(str, enum.Enum):
    pyomo = 'pyomo'
    cpxlp = 'cpxlp'
    nl = 'nl'
    mps = 'mps'
    mod = 'mod'
    lpxlp = 'lpxlp'
    osil = 'osil'
    bar = 'bar'
    gams = 'gams'

    # Overloading __str__ is needed to match the behavior of the old
    # pyutilib.enum class (removed June 2020). There are spots in the
    # code base that expect the string representation for items in the
    # enum to not include the class name. New uses of enum shouldn't
    # need to do this.
    def __str__(self):
        return self.value


#
# osrl - osrl XML file defined by the COIN-OR OS project: Result
# results - A Pyomo results object  (reader define by solver class)
# sol - AMPL *.sol file
# soln - A solver-specific solution file  (reader define by solver class)
# yaml - A Pyomo results file in YAML format
# json - A Pyomo results file in JSON format
#
class ResultsFormat(str, enum.Enum):
    osrl = 'osrl'
    results = 'results'
    sol = 'sol'
    soln = 'soln'
    yaml = 'yaml'
    json = 'json'

    # Overloading __str__ is needed to match the behavior of the old
    # pyutilib.enum class (removed June 2020). There are spots in the
    # code base that expect the string representation for items in the
    # enum to not include the class name. New uses of enum shouldn't
    # need to do this.
    def __str__(self):
        return self.value


def guess_format(filename):
    formats = {}
    formats['py'] = ProblemFormat.pyomo
    formats['nl'] = ProblemFormat.nl
    formats['bar'] = ProblemFormat.bar
    formats['mps'] = ProblemFormat.mps
    formats['mod'] = ProblemFormat.mod
    formats['lp'] = ProblemFormat.cpxlp
    formats['osil'] = ProblemFormat.osil
    formats['gms'] = ProblemFormat.gams
    formats['gams'] = ProblemFormat.gams

    formats['sol'] = ResultsFormat.sol
    formats['osrl'] = ResultsFormat.osrl
    formats['soln'] = ResultsFormat.soln
    formats['yml'] = ResultsFormat.yaml
    formats['yaml'] = ResultsFormat.yaml
    formats['jsn'] = ResultsFormat.json
    formats['json'] = ResultsFormat.json
    formats['results'] = ResultsFormat.yaml
    if filename:
        return formats.get(filename.split('.')[-1].strip(), None)
    else:
        return None
