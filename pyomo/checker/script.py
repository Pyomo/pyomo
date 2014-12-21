#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

class ModelScript(object):

    def __init__(self, filename=None, text=None):
        if filename is not None:
            self._filename = filename
        elif text is not None:
            self._filename = None
            self._text = text
        else:
            raise ValueError("Must provide either a script file or text data")

    def read(self):
        if self._filename is not None:
            with open(self._filename, 'r') as f:
                return f.read()
        else:
            return self._text

    def filename(self):
        if self._filename is not None:
            return self._filename
        else:
            return "<unknown>"
