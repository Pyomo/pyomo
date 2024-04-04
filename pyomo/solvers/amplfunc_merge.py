#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


def amplfunc_string_merge(amplfunc, pyomo_amplfunc):
    """Merge two AMPLFUNC variable strings eliminating duplicate lines"""
    # Assume that the strings amplfunc and pyomo_amplfunc don't contain duplicates
    # Assume that the path separator is correct for the OS so we don't need to
    # worry about comparing Unix and Windows paths.
    amplfunc_lines = amplfunc.split("\n")
    existing = set(amplfunc_lines)
    for line in pyomo_amplfunc.split("\n"):
        # Skip lines we already have
        if line not in existing:
            amplfunc_lines.append(line)
    # Remove empty lines which could happen if one or both of the strings is
    # empty or there are two new lines in a row for whatever reason.
    amplfunc_lines = [s for s in amplfunc_lines if s != ""]
    return "\n".join(amplfunc_lines)


def amplfunc_merge(env):
    """Merge AMPLFUNC and PYOMO_AMPLFUNC in an environment var dict"""
    return amplfunc_string_merge(env.get("AMPLFUNC", ""), env.get("PYOMO_AMPLFUNC", ""))
