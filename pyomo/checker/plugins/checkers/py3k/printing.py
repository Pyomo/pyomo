#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import re
import pyomo.common.plugin
from pyomo.checker.plugins.checker import IterativeDataChecker


class PrintParens(IterativeDataChecker):

    pyomo.common.plugin.alias('py3k.print_parens', 'Check if print statements have parentheses.')

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
