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
import pyomo.common.plugin

from pyomo.checker.plugins.checker import IterativeTreeChecker


class ModelName(IterativeTreeChecker):

    pyomo.common.plugin.alias('model.model_name', 'Check that the "model" variable is assigned with a Pyomo model.')

    def beginChecking(self, runner, script):
        self.modelAssigned = False

    def endChecking(self, runner, script):
        if not self.modelAssigned:
            self.problem("Global object 'model' is never assigned.")

    def checkerDoc(self):
        return """\
        Pyomo will be unable to execute this model
        file without additional arguments.
        """

    def checkTarget(self, node):
        if isinstance(node, ast.Name) and node.id == 'model':
            self.modelAssigned = True

    def check(self, runner, script, info):
        # If assigning, check target name
        if isinstance(info, ast.Assign):
            for target in info.targets:
                self.checkTarget(target)

                # Handle multiple assignment
                if isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        self.checkTarget(elt)


class ModelCreate(IterativeTreeChecker):

    pyomo.common.plugin.alias('model.create', 'Check if a Pyomo model class is being assigned to a variable.')

    def getTargetStrings(self, assign):
        ls = []
        for target in assign.targets:
            if isinstance(target, ast.Name):
                ls.append(target.id)
            elif isinstance(target, ast.Tuple):
                ls.extend(list(map((lambda x: x.id), target.elts))) # TODO probably not resilient
        return ls

    def checkerDoc(self):
        return """\
        Usually, developers create an instance of a Pyomo model class.
        It is rare that the class itself needs to be assigned to a
        variable, e.g.:
            x = ConcreteModel      =>      x = ConcreteModel()
        """

    def check(self, runner, script, info):
        if isinstance(info, ast.Assign):
            if 'model' in self.getTargetStrings(info):
                if isinstance(info.value, ast.Name):
                    if info.value.id in ['Model', 'AbstractModel', 'ConcreteModel']:
                        self.problem("Possible incorrect assignment of " + info.value.id + " class instead of instance", lineno = info.lineno)


class DeprecatedModel(IterativeTreeChecker):

    pyomo.common.plugin.alias('model.Model_class', 'Check if the deprecated Model class is being used.')

    def checkerDoc(self):
        return """\
        The Model class is no longer supported as an object that can be
        created - instead, use the ConcreteModel or AbstractModel class.
        """

    def check(self, runner, script, info):
        if isinstance(info, ast.Assign):
            if isinstance(info.value, ast.Call):
                if isinstance(info.value.func, ast.Name):
                    if info.value.func.id == 'Model':
                        self.problem("Deprecated use of Model class", lineno = info.value.func.lineno)
