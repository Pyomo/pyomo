import ast

from pyomo.core.plugins.check.model import ModelTrackerHook
from pyomo.core.plugins.check.checker import IterativeTreeChecker
from pyomo.core.plugins.check.checkers.pyomo._rulebase import _ModelRuleChecker


class ModelShadowing(IterativeTreeChecker):

    ModelTrackerHook()

    def checkerDoc(self):
        return """\
        It is generally considered poor practice to "shadow", or reuse,
        the name of your model variable in a rule. In your rule definitions,
        consider changing the name of the model argument.
        """

    def check(self, runner, script, info):
        if isinstance(info, ast.FunctionDef):
            for arg in info.args.args:
                if isinstance(arg, ast.Name):
                    if arg.id in script.modelVars:
                        self.problem("Function {0} may shadow model variable {1}".format(info.name, arg.id), lineno=info.lineno)


class ModelAccess(IterativeTreeChecker):

    ModelTrackerHook()

    def checkerDoc(self):
        return """\
        Within model rules, you should access the instance of the model that
        is passed in to the function, rather than the global model instance.
        For example:
            def rule(m, i):
                return m.x[i] >= 10.0 # not model.x[i]
        """

    def check(self, runner, script, info):
        if isinstance(info, ast.FunctionDef):
            attrNodes = [x for x in list(ast.walk(info)) if isinstance(x, ast.Attribute)]
            for attrNode in attrNodes:
                if attrNode.value.id in script.modelVars:
                    self.problem("Expression {0}.{1} may access model outside function scope".format(attrNode.value.id, attrNode.attr), lineno=attrNode.lineno)


class ModelArgument(_ModelRuleChecker):

    def checkerDoc(self):
        return """\
        Model rule functions must have the model as the first argument in
        the function definition. For example, change:
            def con_rule(i, model): # ...
        To:
            def con_rule(model, i): # ...
        """

    def checkBody(self, funcdef):
        for bodyNode in funcdef.body:
            for node in ast.walk(bodyNode):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        if node.value.id != funcdef.args.args[0].id:
                            self.problem("Model argument {0} is not first in rule argument list".format(node.value.id), lineno=funcdef.lineno)


class NoneReturn(_ModelRuleChecker):

    def checkerDoc(self):
        return """\
        Model rule functions may not return None.
        """

    def checkBody(self, funcdef):
        """Look for statements of the format 'return None'"""
        for node in ast.walk(funcdef):
            if isinstance(node, ast.Return):
                if isinstance(node.value, ast.Name):
                    if node.value.id == 'None':
                        self.problem("Cannot return None from model rule {0}".format(funcdef.name), lineno=funcdef.lineno)
