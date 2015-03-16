#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


__all__ = ['TestCase']

import sys
import os
import re
from inspect import getfile

import pyutilib.th as unittest
import pyutilib.subprocess

try:
    import yaml
    using_yaml=True
except ImportError:
    using_yaml=False

def _failIfPyomoResultsDiffer(self, cmd=None, baseline=None, cwd=None):
    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(getfile(self.__class__)))
    oldpwd = os.getcwd()
    os.chdir(cwd)

    try:
        # get the baseline file
        if os.path.exists(baseline):
            INPUT = open(baseline, 'r')
            baseline = "\n".join(INPUT.readlines())
            INPUT.close()
    
        output = pyutilib.subprocess.run(cmd)
    finally:
        os.chdir(oldpwd)
    
    if output[0] != 0:
        self.fail("Command terminated with nonzero status: '%s'" % cmd)
    results = extract_results( re.split('\n',output[1]) )
    try:
        compare_results(results, baseline)
    except IOError:
        err = sys.exc_info()[1]
        self.fail("Command failed to generate results that can be compared with the baseline: '%s'" % err)
    except ValueError:
        err = sys.exc_info()[1]
        self.fail("Difference between results and baseline: '%s'" % err)


class TestCase(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        unittest.TestCase.__init__(self, methodName)

    def failIfPyomoResultsDiffer(self, cmd, baseline, cwd=None):
        if not using_yaml:
            self.fail("Cannot compare Pyomo results because PyYaml is not installed")
        _failIfPyomoResultsDiffer(self, cmd=cmd, baseline=baseline, cwd=cwd)

    @unittest.nottest
    def add_pyomo_results_test(cls, name=None, cmd=None, fn=None, baseline=None, cwd=None):
        if not using_yaml:
            return
        if cmd is None and fn is None:
            print("ERROR: must specify either the 'cmd' or 'fn' option to define how the output file is generated")
            return
        if name is None and baseline is None:
            print("ERROR: must specify a baseline comparison file, or the test name")
            return
        if baseline is None:
            baseline=name+".txt"
        tmp = name.replace("/","_")
        tmp = tmp.replace("\\","_")
        tmp = tmp.replace(".","_")
        #
        # Create an explicit function so we can assign it a __name__ attribute.
        # This is needed by the 'nose' package
        #
        if fn is None:
            func = lambda self,c1=cwd,c2=cmd,c3=tmp+".out",c4=baseline: _failIfPyomoResultsDiffer(self,cwd=c1,cmd=c2,baseline=c4)
        else:
            # This option isn't implemented...
            sys.exit(1)
            func = lambda self,c1=fn,c2=tmp,c3=baseline: _failIfPyomoResultsDiffer(self,fn=c1,name=c2,baseline=c3)
        func.__name__ = "test_"+tmp
        func.__doc__ = "pyomo result test: "+func.__name__+ \
                       " ("+str(cls.__module__)+'.'+str(cls.__name__)+")"
        setattr(cls, "test_"+tmp, func)
    add_pyomo_results_test=classmethod(add_pyomo_results_test)
