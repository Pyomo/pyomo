Importing model and scenario tree files
Importing module=/home/hudson/slave/workspace/Pyomo_trunk_python2.7_snotra_expensive/src/pyomo/examples/pysp/farmer/models/ReferenceModel.py
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
		 QuantityPurchased : [*] 
		 QuantitySubQuotaSold : [*] 
		 QuantitySuperQuotaSold : [*] 
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
Scenario-based instance initialization enabled
Creating instance for scenario=BelowAverageScenario
Data for scenario=BelowAverageScenario loads from file=/home/hudson/slave/workspace/Pyomo_trunk_python2.7_snotra_expensive/src/pyomo/examples/pysp/farmer/scenariodata/BelowAverageScenario.dat
Creating instance for scenario=AverageScenario
Data for scenario=AverageScenario loads from file=/home/hudson/slave/workspace/Pyomo_trunk_python2.7_snotra_expensive/src/pyomo/examples/pysp/farmer/scenariodata/AverageScenario.dat
Creating instance for scenario=AboveAverageScenario
Data for scenario=AboveAverageScenario loads from file=/home/hudson/slave/workspace/Pyomo_trunk_python2.7_snotra_expensive/src/pyomo/examples/pysp/farmer/scenariodata/AboveAverageScenario.dat
Time to construct scenario instances=0.03 seconds
Linking instances into scenario tree
Creating extensive form instance
Writing CVaR weighted objective
CVaR term weight=0.0
CVaR alpha=0.9

Creating variables for master binding instance
Time to construct extensive form instance=0.00 seconds
Starting to write extensive form
Extensive form written to file=/home/hudson/slave/workspace/Pyomo_trunk_python2.7_snotra_expensive/src/pyomo/pyomo/pysp/tests/unit/test_farmer_ef_cvar.lp
Time to write output file=0.02 seconds

Total EF execution time=0.07 seconds

