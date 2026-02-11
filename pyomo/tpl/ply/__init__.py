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

"""PLY
===

This module is derived from the PLY package (3.11) by David Beazley.

It has been modified to disallow loading parse tables from arbitrary
files or pickles.  This simplifies management of the parse table module
and avoids potential security concerns.

The original PLY project (https://github.com/dabeaz/ply) was archived on
December 21, 2025 and is no longer developed or maintained.

"""

__version__ = '3.11.post1'
__all__ = ['lex','yacc']
