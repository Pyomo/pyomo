# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import inspect


def copy_docstrings(reference_class: type, methods: list[str] | None = None):
    """Decorator to copy docstrings from a reference class to the decorated class.

    Note that only docstrings for methods, generators, and functions are
    copied.

    Parameters
    ----------
    reference_class: type
        The source class to copy docstrings from

    methods: list[str] | None
        The list of methods from the `reference_class` to copy
        docstrings from.  If empty or ``None``, then all method
        docstrings are checked / copied.

    """
    if not methods:
        method_list = dir(reference_class)
    else:
        method_list = list(methods)

    def wrapper(cls):
        for method_name in method_list:
            method = getattr(reference_class, method_name)
            if not inspect.isfunction(method) and not inspect.ismethod(method):
                # Skip attributes that are not functions / generators / methods
                continue
            old_doc = getattr(method, '__doc__', None)
            if not old_doc:
                # Skip methods where there isn't a docstring to copy
                continue
            new_method = getattr(cls, method_name, None)
            if new_method is None or getattr(new_method, '__doc__', None):
                # Skip methods that don't exist, or ones that have
                # docstrings defined
                continue
            new_method.__doc__ = old_doc
        return cls

    return wrapper
