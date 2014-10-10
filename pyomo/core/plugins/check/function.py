import ast

from coopr.core.plugin import *
from coopr.pyomo.check.hooks import *


class FunctionTrackerHook(SingletonPlugin):

    implements(IPreCheckHook)

    def precheck(self, runner, script, info):
        # create models dict if nonexistent
        if getattr(script, 'functionDefs', None) is None:
            script.functionDefs = {}

        # add new function definitions
        if isinstance(info, ast.FunctionDef):
            script.functionDefs[info.name] = info

        # update function def dictionary with assignments
        elif isinstance(info, ast.Assign):
            if isinstance(info.value, ast.Name):
                if info.value.id in script.functionDefs:
                    for target in info.targets:
                        if isinstance(target, ast.Name):
                            script.functionDefs[target.id] = script.functionDefs[info.value.id]
            else:
                for target in info.targets:
                    if isinstance(target, ast.Name):
                        if target.id in script.functionDefs:
                            del script.functionDefs[target.id]
