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
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________

import shlex


class Bunch(dict):
    """
    A class that can be used to store a bunch of data dynamically.
    This class allows all other attributes to have a default value of None. 
    This borrows the output formatting ideas from the
    ActiveState Code Container (recipe 496697).
    """

    def __init__(self, *args, **kw):
        for arg in args:
            for item in shlex.split(arg):
                r = item.find('=')
                if r != -1:
                    try:
                        val = eval(item[r + 1:])
                    except:
                        val = item[r + 1:]
                    kw[item[:r]] = val
        dict.__init__(self, kw)
        self.__dict__.update(kw)
        if not '_name_' in kw:
            self._name_ = self.__class__.__name__

    def update(self, d):
        """
        The update is specialized for JSON-like data.  This
        recursively replaces dictionaries with Container objects.
        """
        for k in d:
            if type(d[k]) is dict:
                tmp = Bunch()
                tmp.update(d[k])
                self.__setattr__(k, tmp)
            elif type(d[k]) is list:
                val = []
                for i in d[k]:
                    if type(i) is dict:
                        tmp = Bunch()
                        tmp.update(i)
                        val.append(tmp)
                    else:
                        val.append(i)
                self.__setattr__(k, val)
            else:
                self.__setattr__(k, d[k])

    def set_name(self, name):
        self._name_ = name

    def __setitem__(self, name, val):
        self.__setattr__(name, val)

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setattr__(self, name, val):
        if name[0] != '_':
            dict.__setitem__(self, name, val)
        self.__dict__[name] = val

    def __getattr__(self, name):
        try:
            return dict.__getitem__(self, name)
        except:
            if name[0] == '_':
                raise AttributeError("Unknown attribute %s" % name)
        return None

    def __repr__(self):
        attrs = sorted("%s = %r" % (k, v) for k, v in self.__dict__.items()
                       if not k.startswith("_"))
        return "%s(%s)" % (self.__class__.__name__, ", ".join(attrs))

    def __str__(self):
        return self.as_string()

    def __str__(self, nesting=0, indent=''):
        attrs = []
        indentation = indent + "    " * nesting
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                text = [indentation, k, ":"]
                if isinstance(v, Bunch):
                    if len(v) > 0:
                        text.append('\n')
                    text.append(v.__str__(nesting + 1))
                elif isinstance(v, list):
                    if len(v) == 0:
                        text.append(' []')
                    else:
                        for v_ in v:
                            text.append('\n' + indentation + "-")
                            if isinstance(v_, Bunch):
                                text.append('\n' + v_.__str__(nesting + 1))
                            else:
                                text.append(" " + repr(v_))
                else:
                    text.append(' ' + repr(v))
                attrs.append("".join(text))
        attrs.sort()
        return "\n".join(attrs)
