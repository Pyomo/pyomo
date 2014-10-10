import ast

from coopr.core.plugin import *
from coopr.pyomo.check.hooks import *


class ModelTrackerHook(SingletonPlugin):

    implements(IPreCheckHook)

    def precheck(self, runner, script, info):
        # create models dict if nonexistent
        if getattr(script, 'modelVars', None) is None:
            script.modelVars = {}

        # parse AST node
        if isinstance(info, ast.Assign):
            if isinstance(info.value, ast.Call):
                if isinstance(info.value.func, ast.Name):
                    if info.value.func.id.endswith("Model"):
                        for target in info.targets:
                            if isinstance(target, ast.Name):
                                script.modelVars[target.id] = info.value.func.id
                            elif isinstance(target, ast.Tuple):
                                for elt in target.elts:
                                    if isinstance(elt, ast.Name):
                                        script.modelVars[elt.id] = info.value.func.id
                    else:
                        for target in info.targets:
                            if isinstance(target, ast.Name):
                                if target.id in script.modelVars:
                                    del script.modelVars[target.id]
                            elif isinstance(target, ast.Tuple):
                                for elt in target.elts:
                                    if isinstance(elt, ast.Name):
                                        if elt.id in script.modelVars:
                                            del script.modelVars[elt.id]
