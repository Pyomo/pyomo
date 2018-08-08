#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import networkx

G = networkx.DiGraph()
# first stage
G.add_node("R",
           cost="FirstStageCost",
           variables=["x"],
           derived_variables=["z"])

# second stage
G.add_node("u0",
           cost="SecondStageCost",
           variables=["y0"],
           derived_variables=["xu0"])
G.add_edge("R", "u0", weight=0.1)
G.add_node("u1",
           cost="SecondStageCost",
           variables=["y1"],
           derived_variables=["xu1"])
G.add_edge("R", "u1", weight=0.5)
G.add_node("u2",
           cost="SecondStageCost",
           variables=["y2"],
           derived_variables=["xu2"])
G.add_edge("R", "u2", weight=0.4)

# third stage
G.add_node("u00",
           cost="ThirdStageCost",
           variables=["yu00"])
G.add_edge("u0", "u00", weight=0.1)
G.add_node("u01",
           cost="ThirdStageCost",
           variables=["yu01"])
G.add_edge("u0", "u01", weight=0.9)
G.add_node("u10",
           cost="ThirdStageCost",
           variables=["yu10"])
G.add_edge("u1", "u10", weight=0.5)
G.add_node("u11",
           cost="ThirdStageCost",
           variables=["yu11"])
G.add_edge("u1", "u11", weight=0.5)
G.add_node("u20",
           cost="ThirdStageCost",
           variables=["yu20"])
G.add_edge("u2", "u20", weight=1.0)
