#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

def load():
    import pyomo.scripting.plugins.check
    import pyomo.scripting.plugins.convert
    import pyomo.scripting.plugins.solve
    import pyomo.scripting.plugins.download
    import pyomo.scripting.plugins.build_ext
    import pyomo.scripting.plugins.extras
