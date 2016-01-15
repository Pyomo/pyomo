#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import ast

from pyomo.checker.checker import *
from pyomo.checker.script import ModelScript


class CheckingNodeVisitor(ast.NodeVisitor):
    
    def __init__(self, runner, script, tc = [], dc = [], pt = ""):
        """
        @param tc iterative tree checkers
        @param dc iterative data checkers
        @param pt program text
        """

        super(CheckingNodeVisitor, self).__init__()

        self.runner = runner
        self.script = script
        self.treeCheckers = tc
        self.dataCheckers = dc
        self.programLines = pt.split("\n")
        self.running_lineno = 0

    def sendBegin(self):
        for checker in self.treeCheckers + self.dataCheckers:
            checker._beginChecking(self.runner, self.script)

    def sendEnd(self):
        for checker in self.treeCheckers + self.dataCheckers:
            checker._endChecking(self.runner, self.script)

    def generic_visit(self, node):
        if 'lineno' in dir(node):
            current_lineno = node.lineno
            if current_lineno > self.running_lineno:
                self.running_lineno = current_lineno
                for checker in self.dataCheckers:
                    checker._check(self.runner, self.script, (current_lineno, self.programLines[current_lineno - 1]))

        for checker in self.treeCheckers:
            checker._check(self.runner, self.script, node)

        super(CheckingNodeVisitor, self).generic_visit(node)


class ModelCheckRunner(object):

    _checkers = ExtensionPoint(IModelChecker)

    def __init__(self):
        self.scripts = []

    def run(self, *args, **kwargs):
        from pyomo.checker.plugins.checker import ImmediateDataChecker, IterativeDataChecker, ImmediateTreeChecker, IterativeTreeChecker

        # Get args
        script = kwargs.pop("script", None)
        verbose = kwargs.pop("verbose", False)
        checkers = kwargs.pop("checkers", {})

        # Store args as necessary
        self.verbose = verbose

        # Add script, if given
        if script is not None:
            self.addScript(ModelScript(script))

        # Enable listed checkers
        if checkers == {}:
            print("WARNING: No checkers enabled!")
        for c in self._checkers(all=True):
            if c._checkerPackage() in checkers:
                if c._checkerName() in checkers[c._checkerPackage()]:
                    c.enable()
                else:
                    c.disable()
            else:
                c.disable()

        # Show checkers if requested
        if False:
            printable = {}
            for c in self._checkers():
                if c._checkerPackage() not in printable:
                    printable[c._checkerPackage()] = [c._checkerName()]
                else:
                    printable[c._checkerPackage()].append(c._checkerName())
            
            for package in printable:
                print("{0}: {1}".format(package, " ".join(printable[package])))
            print("")

        # Pre-partition checkers
        immDataCheckers = [c for c in self._checkers if isinstance(c, ImmediateDataChecker)]
        iterDataCheckers = [c for c in self._checkers if isinstance(c, IterativeDataChecker)]
        immTreeCheckers = [c for c in self._checkers if isinstance(c, ImmediateTreeChecker)]
        iterTreeCheckers = [c for c in self._checkers if isinstance(c, IterativeTreeChecker)]

        for script in self.scripts:
            # Read in the script and call data checkers
            data = script.read()
            for checker in immDataCheckers:
                checker._beginChecking(self, script)
                checker._check(self, script, data)
                checker._endChecking(self, script)

            # Get the data into a parse tree
            tree = ast.parse(data)
            for checker in immTreeCheckers:
                checker._beginChecking(self, script)
                checker._check(self, script, tree)
                checker._endChecking(self, script)

            # Start walking the tree, calling checkers along the way
            visitor = CheckingNodeVisitor(self, script, tc=iterTreeCheckers, dc=iterDataCheckers, pt = data)
            visitor.sendBegin()
            visitor.visit(tree)
            visitor.sendEnd()

    def addScript(self, script):
        self.scripts.append(script)
