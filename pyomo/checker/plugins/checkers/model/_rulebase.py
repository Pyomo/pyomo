#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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
