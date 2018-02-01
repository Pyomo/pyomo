import piecewise_scenario_tree

model = piecewise_scenario_tree.model.clone()
model.Scenarios.pprint()
model.Bundling = True
model.Bundles.add("B1")
model.BundleScenarios["B1"] = ["Scenario1", "Scenario2"]
model.Bundles.add("B2")
model.BundleScenarios["B2"] = ["Scenario3"]
