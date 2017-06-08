#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.checker.plugins.checker import IterativeTreeChecker

class PrintASTNodes(IterativeTreeChecker):

    def __init__(self):
        self.disable()

    def check(self, runner, script, info):
        if 'lineno' in dir(info):
            self.problem(str(info), lineno = info.lineno)
        else:
            self.problem(str(info))
