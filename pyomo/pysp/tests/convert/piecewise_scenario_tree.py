from pyomo.pysp.scenariotree.tree_structure_model import \
    CreateConcreteTwoStageScenarioTreeModel

model = CreateConcreteTwoStageScenarioTreeModel(3)

#### helps with testing across python versions
for key in model.ConditionalProbability:
    if key != "RootNode":
        model.ConditionalProbability[key] = 0.3333333333
#####

first_stage = model.Stages.first()
second_stage = model.Stages.last()

# First Stage
model.StageCost[first_stage] = 'StageCost[1]'
model.StageVariables[first_stage].add('x')
model.StageDerivedVariables[first_stage].add('y')
model.StageDerivedVariables[first_stage].add('fx')
model.StageDerivedVariables[first_stage].add('p_first_stage')

# Second Stage
model.StageCost[second_stage] = 'StageCost[2]'
model.StageVariables[second_stage].add('z')
model.StageDerivedVariables[second_stage].add('q')
model.StageDerivedVariables[second_stage].add('fz')
model.StageDerivedVariables[second_stage].add('r')
model.StageDerivedVariables[second_stage].add('p_second_stage')
