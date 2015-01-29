#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

"""
Utilities to support the definition of optimization applications that
can be optimized with the Acro COLIN optimizers.
"""

__all__ = ['AmplModel']

import os

from pyomo.opt.base import (ProblemFormat,
                            convert_problem,
                            guess_format)

class AmplModel(object):
    """
    A class that provides a wrapper for AMPL models.
    """

    def __init__(self, modfile, datfile=None):
        """
        The constructor.
        """
        self.modfile = modfile
        self.datfile = datfile

    def valid_problem_types(self):
        """This method allows the pyomo.opt convert function to work with an AmplModel object."""
        return [ProblemFormat.mod]

    def _problem_files(self):
        if self.datfile is None:
            return [self.modfile]
        else:
            return [self.modfile, self.datfile]

    def write(self, filename, format=None, solver_capability=None):
        """
        Write the model to a file, with a given format.

        NOTE: this is the same exact code as is used in PyomoModel.py
        """
        if format is None and not filename is None:
            #
            # Guess the format if none is specified
            #
            format = guess_format(filename)
        if solver_capability is None:
            solver_capability = lambda x: True
        #
        if self.datfile is None:
            args = (self.modfile,)
        else:
            args = (self.modfile, self.datfile)
        res = convert_problem(args, format, [format], solver_capability)
        if not filename is None:
            os.rename(res[0][0], filename)

