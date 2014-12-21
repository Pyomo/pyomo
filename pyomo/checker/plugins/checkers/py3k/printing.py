#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import re
from pyomo.checker.plugins.checker import IterativeDataChecker


class PrintParens(IterativeDataChecker):

    def __init__(self):
        self.current_lineno = 0

    def check(self, runner, script, info):
        self.current_lineno = info[0]
        line = info[1]
        if re.search("print[^\(]", line) is not None:
            self.problem("Print statements in Python 3.x require parentheses", lineno = info[0])

    def checkerDoc(self):
        return """\
        In Python 3, 'print' changed from a language keyword to a function.
        As such, developers need to surround the arguments to 'print' with
        parentheses, e.g.:
            print "Hello"       =>       print("Hello")
        """
