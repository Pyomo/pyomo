#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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
