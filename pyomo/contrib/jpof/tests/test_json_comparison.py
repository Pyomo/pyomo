#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Test the Pyomo writer for the JSON Parameterized Optimization Format (JPOF)
#

import itertools
import sys
import re
import glob
import os
from os.path import abspath, dirname, join
import json
import pytest

from pyomo.common.fileutils import import_file
from pyomo.common.unittest import assertStructuredAlmostEqual
from pyomo.contrib.jpof import JPOFWriter

currdir = dirname(abspath(__file__))+os.sep
datadir = abspath(join(dirname(dirname(currdir)), 'testCase'))+os.sep


testnames = []
for f in itertools.chain(glob.glob(datadir+'*_testCase.py'),
                         glob.glob(currdir+'*_testCase.py')):
    testnames.append( re.split('[._]',os.path.basename(f))[0] )


@pytest.mark.default
@pytest.mark.parametrize("name", testnames)
def test_jpof_writer(name):
    baseline = currdir+name+'.pyomo.json'
    output = currdir+name+'.test.json'
    if not os.path.exists(baseline):
        pytest.skip("baseline file (%s) not found" % (baseline,))

    if os.path.exists(datadir+name+'_testCase.py'):
        testDir = datadir
    else:
        testDir = currdir
    testCase = import_file(testDir+name+'_testCase.py')
    

    writer = JPOFWriter()
    writer(testCase.model, filename=output, io_options={'file_determinism':3, 'symbolic_solver_labels':True})

    # Check that the pyomo JSON file matches its own baseline
    try:
        with open(baseline, 'r') as f1, open(output, 'r') as f2:
            baseline_contents = json.load(f1)
            output_contents = json.load(f2)
        assertStructuredAlmostEqual(output_contents, baseline_contents,
                                         allow_second_superset=True)
        os.remove(output)
    except Exception:
        err = sys.exc_info()[1]
        pytest.fail("JSON testfile does not match the baseline:\n   testfile="
                  + output + "\n   baseline=" + baseline + "\n" + str(
                      err))

