#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# A simple example illustrating a piecewise
# representation of the step function Z(X)
# 
#          / 0    , 0 <= x <= 1
#  Z(X) >= | 2    , 1 <= x <= 2
#          \ 0.5  , 2 <= x <= 3
#

# **NOTE**: At the domain points 1.0 and 2.0 the
#           range variable can solve to any value
#           on the vertical line. There is no
#           discontinuous "jump".
DOMAIN_PTS = [0., 1., 1., 2., 2., 3.]
RANGE_PTS  = [0., 0., 2., 2., 0.5, 0.5]

from pyomo.core import *

model = ConcreteModel()

model.X = Var(bounds=(0,3))
model.Z = Var()

# See documentation on Piecewise component by typing
# help(Piecewise) in a python terminal after importing pyomo.core
model.con = Piecewise(model.Z,model.X, # range and domain variables
                      pw_pts=DOMAIN_PTS ,
                      pw_constr_type='EQ',
                      f_rule=RANGE_PTS,
                      pw_repn='INC') # **NOTE**: The not all piecewise represenations
                                     #           handle step functions. Those which do
                                     #           not work with step functions are:
                                     #           BIGM_SOS1, BIGM_BIN, and MC

model.obj = Objective(expr=model.Z+model.X, sense=maximize)
