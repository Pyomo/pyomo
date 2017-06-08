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
    import pyomo.opt.plugins.driver
    import pyomo.opt.plugins.colin_xml_io
    import pyomo.opt.plugins.dakota_text_io
    import pyomo.opt.plugins.res
    import pyomo.opt.plugins.sol
    import pyomo.opt.plugins.problem_config

