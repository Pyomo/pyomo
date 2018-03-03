import networkx

G = networkx.DiGraph()
G.add_node("root",
           variables=["x"],
           cost="cost[1]")

G.add_node("s1",
           cost="cost[2]")
G.add_edge("root",
           "s1",
           weight=0.33333333)

G.add_node("s2",
           cost="cost[2]")
G.add_edge("root",
           "s2",
           weight=0.33333334)

G.add_node("s3",
           cost="cost[2]")
G.add_edge("root",
           "s3",
           weight=0.33333333)
