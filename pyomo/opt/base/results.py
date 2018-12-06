#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = [ 'AbstractResultsReader', 'ReaderFactory' ]

from pyomo.common import Factory


ReaderFactory = Factory('problem reader')


class AbstractResultsReader(object):
    """Base class that can read optimization results."""

    def __init__(self, results_format):
        self.format=results_format

    def __call__(self, filename, res=None, suffixes=[]): #pragma:nocover
        raise TypeError("Method __call__ undefined in reader for format "+str(self.format))

    #
    # Support "with" statements. Forgetting to call deactivate
    # on Plugins is a common source of memory leaks
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

