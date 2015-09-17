Importing model and scenario tree files
Importing module=/home/hudson/slave/workspace/Pyomo_trunk_python2.7_snotra_expensive/src/pyomo/examples/pysp/farmerWpiecewise/models/ReferenceModel.py
Module successfully loaded
Scenario Tree Detail
----------------------------------------------------
Tree Nodes:

	Name=AboveAverageNode
	Stage=SecondStage
	Parent=RootNode
	Conditional probability=0.3333
	Children:
		None
	Scenarios:
		AboveAverageScenario

	Name=AverageNode
	Stage=SecondStage
	Parent=RootNode
	Conditional probability=0.3333
	Children:
		None
	Scenarios:
		AverageScenario

	Name=BelowAverageNode
	Stage=SecondStage
	Parent=RootNode
	Conditional probability=0.3333
	Children:
		None
	Scenarios:
		BelowAverageScenario

	Name=RootNode
	Stage=FirstStage
	Parent=None
	Conditional probability=1.0000
	Children:
		AboveAverageNode
		AverageNode
		BelowAverageNode
	Scenarios:
		AboveAverageScenario
		AverageScenario
		BelowAverageScenario

----------------------------------------------------
Stages:
	Name=FirstStage
	Tree Nodes: 
		RootNode
	Variables: 
		 DevotedAcreage : [*] 
	Cost Variable: 
		FirstStageCost

	Name=SecondStage
	Tree Nodes: 
		AboveAverageNode
		AverageNode
		BelowAverageNode
	Variables: 
		 Profit : [*] 
		 QuantityPurchased : [*] 
		 QuantitySold : [*] 
	Cost Variable: 
		SecondStageCost

----------------------------------------------------
Scenarios:
	Name=AboveAverageScenario
	Probability=0.3333
	Leaf node=AboveAverageNode
	Tree node sequence:
		RootNode
		AboveAverageNode

	Name=AverageScenario
	Probability=0.3333
	Leaf node=AverageNode
	Tree node sequence:
		RootNode
		AverageNode

	Name=BelowAverageScenario
	Probability=0.3333
	Leaf node=BelowAverageNode
	Tree node sequence:
		RootNode
		BelowAverageNode

----------------------------------------------------
Scenario tree is valid!
Constructing scenario tree instances
Node-based instance initialization enabled
Creating instance for scenario=BelowAverageScenario
Node data for scenario=BelowAverageScenario partially loading from file=/home/hudson/slave/workspace/Pyomo_trunk_python2.7_snotra_expensive/src/pyomo/examples/pysp/farmerWpiecewise/nodedata/RootNode.dat
Node data for scenario=BelowAverageScenario partially loading from file=/home/hudson/slave/workspace/Pyomo_trunk_python2.7_snotra_expensive/src/pyomo/examples/pysp/farmerWpiecewise/nodedata/BelowAverageNode.dat
Creating instance for scenario=AverageScenario
Node data for scenario=AverageScenario partially loading from file=/home/hudson/slave/workspace/Pyomo_trunk_python2.7_snotra_expensive/src/pyomo/examples/pysp/farmerWpiecewise/nodedata/RootNode.dat
Node data for scenario=AverageScenario partially loading from file=/home/hudson/slave/workspace/Pyomo_trunk_python2.7_snotra_expensive/src/pyomo/examples/pysp/farmerWpiecewise/nodedata/AverageNode.dat
Creating instance for scenario=AboveAverageScenario
Node data for scenario=AboveAverageScenario partially loading from file=/home/hudson/slave/workspace/Pyomo_trunk_python2.7_snotra_expensive/src/pyomo/examples/pysp/farmerWpiecewise/nodedata/RootNode.dat
Node data for scenario=AboveAverageScenario partially loading from file=/home/hudson/slave/workspace/Pyomo_trunk_python2.7_snotra_expensive/src/pyomo/examples/pysp/farmerWpiecewise/nodedata/AboveAverageNode.dat
Time to construct scenario instances=0.07 seconds
Linking instances into scenario tree
Creating extensive form instance
Creating variables for master binding instance
Time to construct extensive form instance=0.00 seconds
Starting to write extensive form
Extensive form written to file=/home/hudson/slave/workspace/Pyomo_trunk_python2.7_snotra_expensive/src/pyomo/pyomo/pysp/tests/unit/test_farmer_piecewise_ef.lp
Time to write output file=0.03 seconds

Total EF execution time=0.13 seconds

