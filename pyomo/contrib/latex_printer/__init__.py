#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Recommended just to build all of the appropriate things
import pyomo.environ

# Remove one layer of .latex_printer
# import statemnt is now:
#   from pyomo.contrib.latex_printer import latex_printer
try:
    from pyomo.contrib.latex_printer.latex_printer import latex_printer
except:
    pass
    # in this case, the dependencies are not installed, nothing will work
