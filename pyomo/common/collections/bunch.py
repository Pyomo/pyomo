#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
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
from collections.abc import Mapping


class Bunch(dict):
    """A class that can be used to store a bunch of data dynamically.

    This class allows for unspecified attributes to have a default value
    of None.  This borrows the output formatting ideas from the
    ActiveState Code Container (recipe 496697).

    For historical reasons, attributes / keys are stored in the
    underlying dict unless they begin with an underscore, in which case
    they are stored as object attributes.

    """

    def __init__(self, *args, **kw):
        self._name_ = self.__class__.__name__
        for arg in args:
            if not isinstance(arg, str):
                raise TypeError("Bunch() positional arguments must be strings")
            for item in shlex.split(arg):
                item = item.split('=', 1)
                if len(item) != 2:
                    raise ValueError(
                        "Bunch() positional arguments must be space separated "
                        f"strings of form 'key=value', got '{item[0]}'"
                    )

                # Historically, this used 'exec'.  That is unsafe in
                # this context (because anyone can pass arguments to a
                # Bunch).  While not strictly backwards compatible,
                # Pyomo was not using this for anything past parsing
                # None/float/int values.  We will explicitly parse those
                # values
                try:
                    val = float(item[1])
                    if int(val) == val:
                        val = int(val)
                    item[1] = val
                except:
                    if item[1].strip() == 'None':
                        item[1] = None
                self[item[0]] = item[1]
        for k, v in kw.items():
            self[k] = v

    def update(self, d):
        """
        The update is specialized for JSON-like data.  This
        recursively replaces dictionaries with Bunch objects.
        """

        def _replace_dict_in_list(lst):
            ans = []
            for v in lst:
                if type(v) is dict:
                    ans.append(Bunch())
                    ans[-1].update(v)
                elif type(v) is list:
                    ans.append(_replace_dict_in_list(v))
                else:
                    ans.append(v)
            return ans

        if isinstance(d, Mapping):
            item_iter = d.items()
        else:
            item_iter = d
        for k, v in item_iter:
            if type(v) is dict:
                self[k] = Bunch()
                self[k].update(v)
            elif type(v) is list:
                self[k] = _replace_dict_in_list(v)
            else:
                self[k] = v

    def set_name(self, name):
        self._name_ = name

    def __getitem__(self, name):
        if not isinstance(name, str):
            raise ValueError(f'Bunch keys must be str (got {type(name).__name__})')
        # Map through Python's standard getattr functionality (which
        # will resolve known attributes without hitting __getattr__)
        return getattr(self, name)

    def __setitem__(self, name, val):
        if not isinstance(name, str):
            raise ValueError(f'Bunch keys must be str (got {type(name).__name__})')
        setattr(self, name, val)

    def __delitem__(self, name):
        if not isinstance(name, str):
            raise ValueError(f'Bunch keys must be str (got {type(name).__name__})')
        delattr(self, name)

    def __getattr__(self, name):
        if name[0] == '_':
            raise AttributeError(f"Unknown attribute '{name}'")
        return self.get(name, None)

    def __setattr__(self, name, val):
        if name[0] == '_':
            super().__setattr__(name, val)
        else:
            super().__setitem__(name, val)

    def __delattr__(self, name):
        if name[0] == '_':
            super().__delattr__(name)
        else:
            super().__delitem__(name)

    def __repr__(self):
        attrs = sorted(
            "%s = %r" % (k, v) for k, v in self.items() if not k.startswith("_")
        )
        return "%s(%s)" % (self.__class__.__name__, ", ".join(attrs))

    def __str__(self, nesting=0, indent=''):
        attrs = []
        indentation = indent + "    " * nesting
        for k, v in self.items():
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
