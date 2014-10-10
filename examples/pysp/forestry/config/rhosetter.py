
def ph_rhosetter_callback(ph, scenario_tree, scenario):

    MyRhoFactorDelta = 0.001
    MyRhoFactorGamma = 0.0001
    MyRhoFactorF = 0.1
    MyRhoFactorZ = 0.01

    scenario_instance = scenario._instance
    symbol_map = scenario_instance._ScenarioTreeSymbolMap
    for tree_node in scenario._node_list:

        stage = tree_node._stage
        t = None
        if stage._name == "Ano1Stage":
            t = "Ano1"
        elif stage._name == "Ano2Stage":
            t = "Ano2"
        elif stage._name == "Ano3Stage":
            t = "Ano3"
        elif stage._name == "Ano4Stage":
            continue
        else:
            raise RuntimeError("Unexpected stage name: "+stage._name)

        for h in scenario_instance.HarvestCells:
            
            new_rho = 0.0
            new_rho += (scenario_instance.P[h,t] * \
                        scenario_instance.A[h] * 
                        MyRhoFactorDelta)
            new_rho += (MyRhoFactorDelta * \
                        scenario_instance.a[h, t] * \
                        scenario_instance.yr[t] * \
                        scenario_instance.A[h] * \
                        sum(scenario_instance.Q[o, t] for o in \
                            scenario_instance.COriginNodeForCell[h]))

            ph.setRhoOneScenario(
                tree_node,
                scenario,
                symbol_map.getSymbol(scenario_instance.delta[h,t]),
                new_rho)
 
            # .a[h, t] * .A[h] * sum([ scenario_instance.Q[o, t] for o in model.OriginNodes for h in model.HCellsForOrigin[o]])
            # scenario_instance.a[h, t] * scenario_instance.A[h] * sum([ scenario_instance.Q[o, t] for o in scenario_instance.COriginNodeForCell[h]])

        for (i,j) in scenario_instance.PotentialRoads:
            ph.setRhoOneScenario(
                tree_node,
                scenario,
                symbol_map.getSymbol(scenario_instance.gamma[i,j,t]),
                scenario_instance.C[i,j,t] * MyRhoFactorGamma)

        for (i,j) in scenario_instance.AllRoads:
            ph.setRhoOneScenario(
                tree_node,
                scenario,
                symbol_map.getSymbol(scenario_instance.f[i,j,t]),
                scenario_instance.D[i,j,t] * MyRhoFactorF)

        for e in scenario_instance.ExitNodes:
            #for e in model.ExitNodes:
            ph.setRhoOneScenario(
                tree_node,
                scenario,
                symbol_map.getSymbol(scenario_instance.z[e,t]),
                scenario_instance.R[e,t] * MyRhoFactorZ)
