for v in model.component_objects(Var):
    print("FOUND VAR:" + v.cname(True))
    v.pprint()

for v_data in model.component_data_objects(Var):
    print("Found: "+v_data.cname(True)+", value = "+str(value(v_data)))
