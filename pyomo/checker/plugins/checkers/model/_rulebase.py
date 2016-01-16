#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import ast

from pyomo.checker.plugins.checker import IterativeTreeChecker
from pyomo.checker.plugins.function import FunctionTrackerHook


class _ModelRuleChecker(IterativeTreeChecker):

    FunctionTrackerHook()

    def check(self, runner, script, info):
        if isinstance(info, ast.Call):
            if hasattr(info.func,'id') and info.func.id in ['Objective', 'Constraint', 'Var', 'Param', 'Set']:
                for keyword in info.keywords:
                    if keyword.arg == 'rule':
                        if isinstance(keyword.value, ast.Name):
                            funcname = keyword.value.id
                            if funcname in script.functionDefs:
                                funcdef = script.functionDefs[funcname]
                                self.checkBody(funcdef)

    def checkBody(self, funcdef):
        pass
