#  ___________________________________________________________________________
# 
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import imp


class _ImportRedirect(object):
    """
    This class redirects imports.

    This is adapted from the _ImportRedirect class defined in the Bottle
    software (see https://github.com/bottlepy/bottle).
    """

    def __init__(self, name, impmask, modulefile, alternates):
        """ Create a virtual package that redirects imports (see PEP 302). """
        self.name = name
        self.impmask = impmask
        self.module = sys.modules.setdefault(name, imp.new_module(name))
        self.module.__dict__.update({
            '__file__': modulefile,
            '__path__': [],
            '__all__': [],
            '__loader__': self,
            '__package__': self.module.__name__
        })
        sys.meta_path.append(self)
        self.alternates = alternates

    def find_module(self, fullname, path=None):
        if '.' not in fullname: return
        packname = fullname.rsplit('.', 1)[0]
        if packname != self.name: return
        return self

    def load_module(self, fullname):
        if fullname in sys.modules: return sys.modules[fullname]
        modname = fullname.rsplit('.', 1)[1]
        if modname in self.alternates:
            realname = self.alternates[modname]
        else:
            realname = self.impmask % modname
        try:
            __import__(realname)
        except ImportError as err:
            print("ERROR importing package '%s'" % fullname)
            print("  Failed to import Pyomo extension package '%s'" % realname)
            print("")
            raise err
        module = sys.modules[fullname] = sys.modules[realname]
        setattr(self.module, modname, module)
        module.__loader__ = self
        return module


"""
Copyright (c) 2017, Marcel Hellkamp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

