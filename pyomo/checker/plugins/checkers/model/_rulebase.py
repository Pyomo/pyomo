import ast

from pyomo.checker.plugins.checker import IterativeTreeChecker
from pyomo.checker.plugins.function import FunctionTrackerHook


class _ModelRuleChecker(IterativeTreeChecker):

    FunctionTrackerHook()

    def check(self, runner, script, info):
        if isinstance(info, ast.Call):
            if info.func.id in ['Objective', 'Constraint', 'Var', 'Param', 'Set']:
                for keyword in info.keywords:
                    if keyword.arg == 'rule':
                        if isinstance(keyword.value, ast.Name):
                            funcname = keyword.value.id
                            if funcname in script.functionDefs:
                                funcdef = script.functionDefs[funcname]
                                self.checkBody(funcdef)

    def checkBody(self, funcdef):
        pass
