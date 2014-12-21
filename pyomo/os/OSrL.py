#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['OSrL']

import os
import sys
import xml
import xml.etree.ElementTree

from pyutilib.enum import Enum
from pyutilib.misc import get_xml_text

#
# A class for reading/writing OSrL file
#
class OSrL(object):

    GeneralStatus = Enum('error', 'warning', 'success', 'normal')
    SolutionStatus = Enum('unbounded', 'globallyOptimal', 'locallyOptimal',
                        'optimal', 'bestSoFar', 'feasible', 'infeasible',
                        'stoppedByLimit', 'unsure', 'error', 'other')

    def __init__(self):
        self.solution = []
        self.etree = None
        self.namespace = 'os.optimizationservices.org'

    def write(self,ostream=None, prefix=""):
        if ostream is None:                     #pragma:nocover
            ostream = sys.stdout
        if type(ostream) is str:
            self._tmpfile = ostream
            ostream = open(self._tmpfile,"w")
        else:
            self._tmpfile = None
        self.etree.write(ostream.fileno())

    def read(self, filename):
        self.etree = xml.etree.ElementTree.parse(filename)
        self.validate()

    def validate(self):
        if self.etree is None:                  #pragma:nocover
            return
        doc = self.etree.getroot()
        e = doc.find('.//{%s}general' % self.namespace)
        e = e.find('.//{%s}generalStatus' % self.namespace)
        if e.attrib['type'] not in OSrL.GeneralStatus:
            raise ValueError("General status value '%s' is not in OSrL.GeneralStatus" % str(e.attrib['type']))
        #
        for soln in doc.findall('.//{%s}solution' % self.namespace):
            e = soln.find('.//{%s}status' % self.namespace)
            if e.attrib['type'] not in OSrL.SolutionStatus:
                raise ValueError("Solution status value '%s' is not in OSrL.SolutionStatus" % str(e.attrib['type']))

