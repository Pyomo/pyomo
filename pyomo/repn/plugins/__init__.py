#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

def load():
    import pyomo.repn.plugins.cpxlp
    import pyomo.repn.plugins.ampl
    import pyomo.repn.plugins.baron_writer
    import pyomo.repn.plugins.mps

