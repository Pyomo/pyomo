#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import ast
import pyomo.util.plugin

from pyomo.checker.plugins.model import ModelTrackerHook
from pyomo.checker.plugins.checkers.model._rulebase import _ModelRuleChecker


class ModelValue(_ModelRuleChecker):

    pyomo.util.plugin.alias('model.value', 'Check if comparisons are done using the "value()" function.')

    ModelTrackerHook()
    
    def checkerDoc(self):
        return """\
        Comparisons done on model objects should generally be wrapped in
        a call to value(). The comparison alone will not produce a True/False
        result, but instead generate an expression for later use in a model.
        """

    def check(self, runner, script, info):
        # call superclass to execute checkBody() as necessary
        _ModelRuleChecker.check(self, runner, script, info)

        # also check global If statements
        if isinstance(info, ast.If):
            self.checkCompare(info.test, script = script)

    def checkBody(self, funcdef):
        """Check the body of a function definition for model comparisons local
           to its scope (i.e. using its model argument)."""

        if not isinstance(funcdef.args.args[0], ast.Name):
            return
        modelArg = funcdef.args.args[0].id

        for bodyNode in funcdef.body:
            for node in ast.walk(bodyNode):
                if isinstance(node, ast.If):
                    self.checkCompare(node.test, modelName = modelArg)

    def checkCompare(self, compare, modelName = None, script = None):
        """Check an AST Compare node - iterate for Attribute nodes and match
           against modelName argument. Recurse for script's model defs."""

        if modelName is None and script is None:
            return
        
        if modelName is not None:
            valueCallArgs = []
            generatorExps = []
            for node in ast.walk(compare):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        if node.value.id == modelName:
                            wrapped = self.checkWrapped(node, valueCallArgs, generatorExps)
                            if not wrapped:
                                self.problem("Comparison on attribute {0}.{1} not wrapped in value()".format(modelName, node.attr), lineno=compare.lineno)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id == 'value':
                            valueCallArgs.append(node.args)
                elif isinstance(node, ast.GeneratorExp):
                    generatorExps.append(node)

        if script is not None:
            for name in script.modelVars:
                self.checkCompare(compare, modelName = name)

    def checkWrapped(self, attrNode, valueCallArgs, generatorExps):
        """check if the given attribute node has been 'wrapped', either
           in a value() call or as part of the iterator in a generator
           expression"""
        for i in range(len(valueCallArgs)):
            for j in range(len(valueCallArgs[i])):
                # i = call idx (to return), j = arg idx
                argNode = valueCallArgs[i][j]
                for subnode in ast.walk(argNode):
                    if subnode is attrNode:
                        return True
        for genExp in generatorExps:
            for generator in genExp.generators:
                for subnode in ast.walk(generator.iter):
                    if subnode is attrNode:
                        return True
        return False
