from pyomo.environ import *

model = AbstractModel()
model.x = Var()
model.p = Param(mutable=True, initialize=1.0)
def cost_rule(model, i):
    if i == 1:
        return model.x
    else:
        return 0.0
model.cost = Expression([1,2], rule=cost_rule)
def o_rule(model):
    return model.x
model.o = Objective(rule=o_rule)
def c_rule(model):
    return model.x >= model.p
model.c = Constraint(rule=c_rule)

def pysp_instance_creation_callback(scenario_tree, scenario_name, node_names):
    instance = model.create_instance()
    if scenario_name == "s1":
        instance.p.value = 1.0
    elif scenario_name == "s2":
        instance.p.value = 2.0
    else:
        assert scenario_name == "s3"
        instance.p.value = 3.0
    return instance

def pysp_scenario_tree_model_callback():
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
    return G
