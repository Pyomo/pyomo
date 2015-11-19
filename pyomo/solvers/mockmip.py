#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


import glob
import shutil
from os.path import join, basename, dirname, isfile

from pyomo.opt.base.solvers import _extract_version

class MockMIP(object):
    """Methods used to create a mock MIP solver used for testing
    """

    def __init__(self, mockdir):
        self.mock_subdir=mockdir

    def create_command_line(self,executable,problem_files):
        self._mock_problem = basename(problem_files[0]).split('.')[0]
        self._mock_dir = dirname(problem_files[0])

    def _default_executable(self):
        return "mock"
    executable = _default_executable

    def version(self):
        return _extract_version('')

    def _execute_command(self,cmd):
        mock_basename = join(self._mock_dir, self.mock_subdir, self._mock_problem)
        if self._soln_file is not None:
            # prefer .sol over .soln
            for ext in ('sol', 'soln'):
                file = glob.glob(mock_basename + "*." + ext)
                if len(file):
                    if len(file) > 1:
                        raise RuntimeError("Multiple .%s files found" % ext)
                    shutil.copyfile(file[0], self._soln_file)
                    break
        for file in glob.glob(mock_basename + "*"):
            if file.split(".")[-1] != "out":
                shutil.copyfile(file, join(self._mock_dir, basename(file)))
        log=""
        fname = mock_basename + ".out"
        if not isfile(fname):
            raise ValueError("Missing mock data file: "+fname)
        INPUT=open(mock_basename + ".out")
        for line in INPUT:
            log = log+line
        INPUT.close()
        return [0,log]
