#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.util.plugin import *


class IModelChecker(Interface):

    def check(self, runner, script, info):
        """
        Check a particular piece of Python information for errors.

        Provides the primary interface for checking some code for problems,
        according to the particular subclass's definition. 
        
        @param runner the ModelCheckRunner instance that has dispatched
                      this call to check().
        @param script the ModelScript instance being checked.
        @param info the data to check. Depending on the subclass, info
                    can be the raw text of a Python script, the entire AST
                    of the script, or a particular node in that AST.
        """

    def beginChecking(self, runner, script):
        """
        Start checking the given script from the given runner.
        """

    def endChecking(self, runner, script):
        """
        Finish checking the given script from the given runner.
        """

    def problem(self, script = None, message = "Error", lineno = None):
        """
        Write a problem to the console. The format varies and can be
        changed in subclasses; by default, this method prints the following:

        [CheckerName] script.py:line: Error

        @param script the ModelScript instance being checked. Must be passed
                      to have the file name and line number printed.
        @param message the error to display.
        @param lineno the line number on which the error occurred.
        """

