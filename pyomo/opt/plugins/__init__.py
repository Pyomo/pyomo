#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

def load():
    import pyomo.opt.plugins.driver
    import pyomo.opt.plugins.colin_xml_io
    import pyomo.opt.plugins.dakota_text_io
    import pyomo.opt.plugins.res
    import pyomo.opt.plugins.sol
    import pyomo.opt.plugins.problem_config

