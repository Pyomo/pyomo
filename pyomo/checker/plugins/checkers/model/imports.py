#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import ast

from pyomo.checker.plugins.checker import IterativeTreeChecker


class Imports(IterativeTreeChecker):
    """
    Check that an import for the pyomo.core package exists
    somewhere within the initial imports block
    """

    def beginChecking(self, runner, script):
        self.pyomoImported = False

    def endChecking(self, runner, script):
        if not self.pyomoImported:
            self.problem("The model script never imports pyomo.core.")

    def checkerDoc(self):
        return """\
        You may have trouble creating model components.
        Consider adding the following statement at the
        top of your model file:
            from pyomo.core import *
        """

    def check(self, runner, script, info):
        if isinstance(info, ast.Import):
            for name in info.names:
                if isinstance(name, ast.alias):
                    if name.name == 'pyomo.core':
                        self.pyomoImported = True

        if isinstance(info, ast.ImportFrom):
            if info.module == 'pyomo.core':
                self.pyomoImported = True
