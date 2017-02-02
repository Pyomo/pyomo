#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import sys
import ast
import pyomo.util.plugin

from pyomo.checker.plugins.model import ModelTrackerHook
from pyomo.checker.plugins.checker import IterativeTreeChecker
from pyomo.checker.plugins.checkers.model._rulebase import _ModelRuleChecker


if sys.version_info < (3,0):
    def arg_name(arg):
        return arg.id
else:   
    def arg_name(arg):
        return arg.arg


if False:
  # WEH: I don't think we should complain about this.

  class ModelShadowing(IterativeTreeChecker):

    pyomo.util.plugin.alias('model.rule.shadowing', 'Ignoring for now')

    ModelTrackerHook()

    def checkerDoc(self):
        return """\
        Reusing the name of your model variable in a rule may lead to problems where
        the variable shadows the global value.  In your rule definitions,
        consider changing the name of the model argument.
        """

    def check(self, runner, script, info):
        if isinstance(info, ast.FunctionDef):
            for arg in info.args.args:
                if isinstance(arg, ast.Name):
                    if arg.id in script.modelVars:
                        self.problem("Function {0} may shadow model variable {1}".format(info.name, arg.id), lineno=info.lineno)


class ModelAccess(IterativeTreeChecker):

    pyomo.util.plugin.alias('model.rule.model_access', 'Check that a rule does not reference a global model instance.')

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
                    args = getattr(script, 'functionArgs', [])
                    if len(args) > 0 and not attrNode.value.id in list(arg_name(arg) for arg in args[-1].args):
                        # NOTE: this probably will not catch arguments defined as keyword arguments.
                        self.problem("Expression '{0}.{1}' may access a model variable that is outside of the function scope".format(attrNode.value.id, attrNode.attr), lineno=attrNode.lineno)


class ModelArgument(_ModelRuleChecker):

    pyomo.util.plugin.alias('model.rule.model_argument', 'Check that the model instance is the first argument for a rule.')

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
                        if node.value.id != arg_name(funcdef.args.args[0]):
                            self.problem("Model variable '{0}' is used in the rule, but this variable is not first argument in the rule argument list".format(node.value.id), lineno=funcdef.lineno)


class NoneReturn(_ModelRuleChecker):

    pyomo.util.plugin.alias('model.rule.none_return', 'Check that a rule does not return the value None.')

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
