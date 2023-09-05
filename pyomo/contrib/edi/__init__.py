#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Recommended just to build all of the appropriate things
import pyomo.environ

# Import the relevant classes from Formulation
try:
    from pyomo.contrib.edi.formulation import Formulation
except:
    pass
    # in this case, the dependencies are not installed, nothing will work


# Import the black box modeling tools
try:
    from pyomo.contrib.edi.blackBoxFunctionModel import BlackBoxFunctionModel
    from pyomo.contrib.edi.blackBoxFunctionModel import (
        BlackBoxFunctionModel_Variable as BlackBoxVariable,
    )
    from pyomo.contrib.edi.blackBoxFunctionModel import (
        BlackBoxFunctionModel_Variable as BBVariable,
    )
    from pyomo.contrib.edi.blackBoxFunctionModel import (
        BlackBoxFunctionModel_Variable as BBV,
    )
except:
    pass
    # in this case, the dependencies are not installed, nothing will work
