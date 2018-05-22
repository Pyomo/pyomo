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


class XRange(IterativeTreeChecker):

    pyomo.common.plugin.alias('py3k.xrange', 'Check if the xrange() function is used.')

    def check(self, runner, script, info):
        if isinstance(info, ast.Name):
            if info.id == 'xrange':
                self.problem("'xrange' function was removed in Python 3.")

    def checkerDoc(self):
        return """\
        In Python 3, 'xrange' was removed in favor of 'range', which was
        reimplemented more efficiently. Please change your uses of 'xrange'
        into 'range', e.g.:
            xrange(1,10)       =>       range(1,10)
        """
