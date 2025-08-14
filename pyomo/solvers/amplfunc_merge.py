#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.deprecation import relocated_module_attribute

relocated_module_attribute(
    'amplfunc_string_merge',
    'pyomo.solvers.amplfunc_merge.unique_paths',
    version='6.9.2',
    f_globals=globals(),
)


def unique_paths(*paths):
    """Merge paths, eliminating duplicates.

    Parameters
    ----------
    *path : str

        Each argument is a string containing one or more paths.
        Multiple libraries are separated by newlines.

    Returns
    -------
    str : merged list of newline-separated paths.

    """
    funcs = {}
    for src in paths:
        for line in src.splitlines():
            if not line:
                continue
            funcs[line] = None
    return "\n".join(funcs)


def amplfunc_merge(env, *funcs):
    """Merge AMPLFUNC and PYOMO_AMPLFUNC environment variables with provided values

    Paths are returned with entries from AMPLFUNC first, PYOMO_AMPLFUNC
    second, and the user-provided arguments last.  Duplicates and empty
    paths are filtered out.

    Parameters
    ----------
    env : Dict[str, str]
        Environment dictionary mapping environment variable names to values.

    *funcs : str
        Additional paths to combine.

    """
    return unique_paths(env.get("AMPLFUNC", ""), env.get("PYOMO_AMPLFUNC", ""), *funcs)
