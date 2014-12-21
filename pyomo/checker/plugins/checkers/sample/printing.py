#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.checker.plugins.checker import IterativeTreeChecker

class PrintASTNodes(IterativeTreeChecker):

    def __init__(self):
        self.disable()

    def check(self, runner, script, info):
        if 'lineno' in dir(info):
            self.problem(str(info), lineno = info.lineno)
        else:
            self.problem(str(info))
