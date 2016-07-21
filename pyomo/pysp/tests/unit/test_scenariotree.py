#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
import pyutilib.th as unittest

from pyomo.pysp.scenariotree.tree_structure_model import \
    (ScenarioTreeModelFromNetworkX,
     CreateConcreteTwoStageScenarioTreeModel)
from pyomo.pysp.scenariotree.tree_structure import ScenarioTree
from pyomo.core import (ConcreteModel,
                        Set,
                        Var,
                        Expression,
                        Objective,
                        Block)

try:
    import networkx
    has_networkx = True
except:
    has_networkx = False

class TestScenarioTree(unittest.TestCase):

    def _get_block_model(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1,2])
        b = Block(concrete=True)
        b.s = Set(initialize=[1,2])
        b.x = Var()
        b.X = Var(model.s)
        model.b1 = b.clone()
        model.b2 = b.clone()
        model.b3 = b.clone()
        model.b4 = b.clone()
        model.B1 = Block(model.s, rule=lambda _,i: b.clone())
        model.B2 = Block(model.s, rule=lambda _,i: b.clone())
        model.B3 = Block(model.s, rule=lambda _,i: b.clone())
        model.B4 = Block(model.s, rule=lambda _,i: b.clone())
        model.FirstStageCost = Expression(expr=0.0)
        model.SecondStageCost = Expression(expr=0.0)
        model.obj = Objective(expr=0.0)
        return model

    def test_indexedblock_noindextemplate(self):
        st_model = CreateConcreteTwoStageScenarioTreeModel(1)
        st_model.StageVariables['Stage1'].add("B1")
        st_model.StageDerivedVariables['Stage1'].add("B2")
        st_model.NodeVariables['RootNode'].add("B3")
        st_model.NodeDerivedVariables['RootNode'].add("B4")
        st_model.StageCost['Stage1'] = "FirstStageCost"
        st_model.StageCost['Stage2'] = "SecondStageCost"

        scenario_tree = ScenarioTree(scenariotreeinstance=st_model)
        self.assertEqual(len(scenario_tree.stages), 2)
        self.assertEqual(len(scenario_tree.nodes), 2)
        self.assertEqual(len(scenario_tree.scenarios), 1)

        model = self._get_block_model()

        scenario_tree.linkInInstances({'Scenario1': model})

        root = scenario_tree.findRootNode()
        self.assertEqual(len(root._variable_ids), 24)
        self.assertEqual(len(root._standard_variable_ids), 12)
        self.assertEqual(len(root._derived_variable_ids), 12)
        for name in ("B1[1].x", "B1[2].x",
                     "B2[1].x", "B2[2].x",
                     "B3[1].x", "B3[2].x",
                     "B4[1].x", "B4[2].x"):
            for index in [None]:
                self.assertEqual(
                    (name,index) in root._name_index_to_id, True)
        for name in ("B1[1].X", "B1[2].X",
                     "B2[1].X", "B2[2].X",
                     "B3[1].X", "B3[2].X",
                     "B4[1].X", "B4[2].X"):
            for index in model.s:
                self.assertEqual(
                    (name,index) in root._name_index_to_id, True)

    def test_indexedblock_wildcardtemplate(self):
        st_model = CreateConcreteTwoStageScenarioTreeModel(1)
        st_model.StageVariables['Stage1'].add("B1[*]")
        st_model.StageDerivedVariables['Stage1'].add("B2[*]")
        st_model.NodeVariables['RootNode'].add("B3[*]")
        st_model.NodeDerivedVariables['RootNode'].add("B4[*]")
        st_model.StageCost['Stage1'] = "FirstStageCost"
        st_model.StageCost['Stage2'] = "SecondStageCost"

        scenario_tree = ScenarioTree(scenariotreeinstance=st_model)
        self.assertEqual(len(scenario_tree.stages), 2)
        self.assertEqual(len(scenario_tree.nodes), 2)
        self.assertEqual(len(scenario_tree.scenarios), 1)

        model = self._get_block_model()

        scenario_tree.linkInInstances({'Scenario1': model})

        root = scenario_tree.findRootNode()
        self.assertEqual(len(root._variable_ids), 24)
        self.assertEqual(len(root._standard_variable_ids), 12)
        self.assertEqual(len(root._derived_variable_ids), 12)
        for name in ("B1[1].x", "B1[2].x",
                     "B2[1].x", "B2[2].x",
                     "B3[1].x", "B3[2].x",
                     "B4[1].x", "B4[2].x"):
            for index in [None]:
                self.assertEqual(
                    (name,index) in root._name_index_to_id, True)
        for name in ("B1[1].X", "B1[2].X",
                     "B2[1].X", "B2[2].X",
                     "B3[1].X", "B3[2].X",
                     "B4[1].X", "B4[2].X"):
            for index in model.s:
                self.assertEqual(
                    (name,index) in root._name_index_to_id, True)

    def test_singletonblock_wildcardtemplate(self):
        st_model = CreateConcreteTwoStageScenarioTreeModel(1)
        st_model.StageVariables['Stage1'].add("b1[*]")
        st_model.StageDerivedVariables['Stage1'].add("b2[*]")
        st_model.NodeVariables['RootNode'].add("b3[*]")
        st_model.NodeDerivedVariables['RootNode'].add("b4[*]")
        st_model.StageCost['Stage1'] = "FirstStageCost"
        st_model.StageCost['Stage2'] = "SecondStageCost"

        scenario_tree = ScenarioTree(scenariotreeinstance=st_model)
        self.assertEqual(len(scenario_tree.stages), 2)
        self.assertEqual(len(scenario_tree.nodes), 2)
        self.assertEqual(len(scenario_tree.scenarios), 1)

        model = self._get_block_model()

        scenario_tree.linkInInstances({'Scenario1': model})

        root = scenario_tree.findRootNode()
        self.assertEqual(len(root._variable_ids), 12)
        self.assertEqual(len(root._standard_variable_ids), 6)
        self.assertEqual(len(root._derived_variable_ids), 6)
        for name in ("b1.x", "b2.x", "b3.x", "b4.x"):
            for index in [None]:
                self.assertEqual(
                    (name,index) in root._name_index_to_id, True)
        for name in ("b1.X", "b2.X", "b3.X", "b4.X"):
            for index in model.s:
                self.assertEqual(
                    (name,index) in root._name_index_to_id, True)

    def test_singletonblock_noindextemplate(self):
        st_model = CreateConcreteTwoStageScenarioTreeModel(1)
        st_model.StageVariables['Stage1'].add("b1")
        st_model.StageDerivedVariables['Stage1'].add("b2")
        st_model.NodeVariables['RootNode'].add("b3")
        st_model.NodeDerivedVariables['RootNode'].add("b4")
        st_model.StageCost['Stage1'] = "FirstStageCost"
        st_model.StageCost['Stage2'] = "SecondStageCost"

        scenario_tree = ScenarioTree(scenariotreeinstance=st_model)
        self.assertEqual(len(scenario_tree.stages), 2)
        self.assertEqual(len(scenario_tree.nodes), 2)
        self.assertEqual(len(scenario_tree.scenarios), 1)

        model = self._get_block_model()

        scenario_tree.linkInInstances({'Scenario1': model})

        root = scenario_tree.findRootNode()
        self.assertEqual(len(root._variable_ids), 12)
        self.assertEqual(len(root._standard_variable_ids), 6)
        self.assertEqual(len(root._derived_variable_ids), 6)
        for name in ("b1.x", "b2.x", "b3.x", "b4.x"):
            for index in [None]:
                self.assertEqual(
                    (name,index) in root._name_index_to_id, True)
        for name in ("b1.X", "b2.X", "b3.X", "b4.X"):
            for index in model.s:
                self.assertEqual(
                    (name,index) in root._name_index_to_id, True)

    def test_singletonvar_noindextemplate(self):
        st_model = CreateConcreteTwoStageScenarioTreeModel(1)
        st_model.StageVariables['Stage1'].add("x")
        st_model.StageDerivedVariables['Stage1'].add("y")
        st_model.NodeVariables['RootNode'].add("z")
        st_model.NodeDerivedVariables['RootNode'].add("q")
        st_model.StageCost['Stage1'] = "FirstStageCost"
        st_model.StageCost['Stage2'] = "SecondStageCost"

        scenario_tree = ScenarioTree(scenariotreeinstance=st_model)
        self.assertEqual(len(scenario_tree.stages), 2)
        self.assertEqual(len(scenario_tree.nodes), 2)
        self.assertEqual(len(scenario_tree.scenarios), 1)

        model = ConcreteModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.q = Var()
        model.FirstStageCost = Expression(expr=0.0)
        model.SecondStageCost = Expression(expr=0.0)
        model.obj = Objective(expr=0.0)

        scenario_tree.linkInInstances({'Scenario1': model})

        root = scenario_tree.findRootNode()
        self.assertEqual(len(root._variable_ids), 4)
        self.assertEqual(len(root._standard_variable_ids), 2)
        self.assertEqual(len(root._derived_variable_ids), 2)
        for name in ("x", "y", "z", "q"):
            for index in [None]:
                self.assertEqual(
                    (name,index) in root._name_index_to_id, True)

    def test_singletonvar_wildcardtemplate(self):
        st_model = CreateConcreteTwoStageScenarioTreeModel(1)
        st_model.StageVariables['Stage1'].add("x[*]")
        st_model.StageDerivedVariables['Stage1'].add("y[*]")
        st_model.NodeVariables['RootNode'].add("z[*]")
        st_model.NodeDerivedVariables['RootNode'].add("q[*]")
        st_model.StageCost['Stage1'] = "FirstStageCost"
        st_model.StageCost['Stage2'] = "SecondStageCost"

        scenario_tree = ScenarioTree(scenariotreeinstance=st_model)
        self.assertEqual(len(scenario_tree.stages), 2)
        self.assertEqual(len(scenario_tree.nodes), 2)
        self.assertEqual(len(scenario_tree.scenarios), 1)

        model = ConcreteModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.q = Var()
        model.FirstStageCost = Expression(expr=0.0)
        model.SecondStageCost = Expression(expr=0.0)
        model.obj = Objective(expr=0.0)

        scenario_tree.linkInInstances({'Scenario1': model})

        root = scenario_tree.findRootNode()
        self.assertEqual(len(root._variable_ids), 4)
        self.assertEqual(len(root._standard_variable_ids), 2)
        self.assertEqual(len(root._derived_variable_ids), 2)
        for name in ("x", "y", "z", "q"):
            for index in [None]:
                self.assertEqual(
                    (name,index) in root._name_index_to_id, True)

    def test_multiindexedvar_singlewildcardtemplate(self):
        st_model = CreateConcreteTwoStageScenarioTreeModel(1)
        st_model.StageVariables['Stage1'].add("x[*,* ]")
        st_model.StageDerivedVariables['Stage1'].add("y[ *,*]")
        st_model.NodeVariables['RootNode'].add("z[*,*]")
        st_model.NodeDerivedVariables['RootNode'].add("q[ * , * ]")
        st_model.StageCost['Stage1'] = "FirstStageCost"
        st_model.StageCost['Stage2'] = "SecondStageCost"

        scenario_tree = ScenarioTree(scenariotreeinstance=st_model)
        self.assertEqual(len(scenario_tree.stages), 2)
        self.assertEqual(len(scenario_tree.nodes), 2)
        self.assertEqual(len(scenario_tree.scenarios), 1)

        model = ConcreteModel()
        model.s = Set(initialize=[(1,'a'),(2,'b'),(3,'c')])
        model.x = Var(model.s)
        model.y = Var(model.s)
        model.z = Var(model.s)
        model.q = Var(model.s)
        model.FirstStageCost = Expression(expr=0.0)
        model.SecondStageCost = Expression(expr=0.0)
        model.obj = Objective(expr=0.0)

        scenario_tree.linkInInstances({'Scenario1': model})

        root = scenario_tree.findRootNode()
        self.assertEqual(len(root._variable_ids), 12)
        self.assertEqual(len(root._standard_variable_ids), 6)
        self.assertEqual(len(root._derived_variable_ids), 6)
        for name in ("x", "y", "z", "q"):
            for index in model.s:
                self.assertEqual(
                    (name,index) in root._name_index_to_id, True)

    def test_indexedvar_indextemplate(self):
        st_model = CreateConcreteTwoStageScenarioTreeModel(1)
        st_model.StageVariables['Stage1'].add("x[*]")
        st_model.StageDerivedVariables['Stage1'].add("y[*]")
        st_model.NodeVariables['RootNode'].add("z[*]")
        st_model.NodeDerivedVariables['RootNode'].add("q[*]")
        st_model.StageCost['Stage1'] = "FirstStageCost"
        st_model.StageCost['Stage2'] = "SecondStageCost"

        scenario_tree = ScenarioTree(scenariotreeinstance=st_model)
        self.assertEqual(len(scenario_tree.stages), 2)
        self.assertEqual(len(scenario_tree.nodes), 2)
        self.assertEqual(len(scenario_tree.scenarios), 1)

        model = ConcreteModel()
        model.s = Set(initialize=[1,2,3])
        model.x = Var(model.s)
        model.y = Var(model.s)
        model.z = Var(model.s)
        model.q = Var(model.s)
        model.FirstStageCost = Expression(expr=0.0)
        model.SecondStageCost = Expression(expr=0.0)
        model.obj = Objective(expr=0.0)

        scenario_tree.linkInInstances({'Scenario1': model})

        root = scenario_tree.findRootNode()
        self.assertEqual(len(root._variable_ids), 12)
        self.assertEqual(len(root._standard_variable_ids), 6)
        self.assertEqual(len(root._derived_variable_ids), 6)
        for name in ("x", "y", "z", "q"):
            for index in model.s:
                self.assertEqual(
                    (name,index) in root._name_index_to_id, True)

    def test_indexedvar_noindextemplate(self):
        st_model = CreateConcreteTwoStageScenarioTreeModel(1)
        st_model.StageVariables['Stage1'].add("x")
        st_model.StageDerivedVariables['Stage1'].add("y")
        st_model.NodeVariables['RootNode'].add("z")
        st_model.NodeDerivedVariables['RootNode'].add("q")
        st_model.StageCost['Stage1'] = "FirstStageCost"
        st_model.StageCost['Stage2'] = "SecondStageCost"

        scenario_tree = ScenarioTree(scenariotreeinstance=st_model)
        self.assertEqual(len(scenario_tree.stages), 2)
        self.assertEqual(len(scenario_tree.nodes), 2)
        self.assertEqual(len(scenario_tree.scenarios), 1)

        model = ConcreteModel()
        model.s = Set(initialize=[1,2,3])
        model.x = Var(model.s)
        model.y = Var(model.s)
        model.z = Var(model.s)
        model.q = Var(model.s)
        model.FirstStageCost = Expression(expr=0.0)
        model.SecondStageCost = Expression(expr=0.0)
        model.obj = Objective(expr=0.0)

        scenario_tree.linkInInstances({'Scenario1': model})

        root = scenario_tree.findRootNode()
        self.assertEqual(len(root._variable_ids), 12)
        self.assertEqual(len(root._standard_variable_ids), 6)
        self.assertEqual(len(root._derived_variable_ids), 6)
        for name in ("x", "y", "z", "q"):
            for index in model.s:
                self.assertEqual(
                    (name,index) in root._name_index_to_id, True)

@unittest.skipIf(not has_networkx, "Requires networkx module")
class TestScenarioTreeFromNetworkX(unittest.TestCase):

    def test_empty(self):
        G = networkx.DiGraph()
        with self.assertRaises(networkx.NetworkXPointlessConcept):
            ScenarioTreeModelFromNetworkX(G)

    def test_not_tree(self):
        G = networkx.DiGraph()
        G.add_node("1")
        G.add_node("2")
        G.add_edge("1", "2")
        G.add_edge("2", "1")
        with self.assertRaises(TypeError):
            ScenarioTreeModelFromNetworkX(G)

    def test_not_branching(self):
        G = networkx.DiGraph()
        G.add_node("1")
        G.add_node("2")
        G.add_node("R")
        G.add_edge("1", "R")
        G.add_edge("2", "R")
        with self.assertRaises(TypeError):
            ScenarioTreeModelFromNetworkX(G)

    def test_not_enough_stages(self):
        G = networkx.DiGraph()
        G.add_node("R")
        with self.assertRaises(ValueError):
            ScenarioTreeModelFromNetworkX(G)

    def test_missing_name(self):
        G = networkx.DiGraph()
        G.add_node("R", name="Root")
        G.add_node("C")
        G.add_edge("R", "C", probability=1)
        with self.assertRaises(KeyError):
            ScenarioTreeModelFromNetworkX(
                G,
                node_name_attribute="name")

    def test_missing_probability(self):
        G = networkx.DiGraph()
        G.add_node("R", name="Root")
        G.add_node("C", name="Child")
        G.add_edge("R", "C")
        with self.assertRaises(KeyError):
            ScenarioTreeModelFromNetworkX(G)

    def test_bad_probability1(self):
        G = networkx.DiGraph()
        G.add_node("R",)
        G.add_node("C",)
        G.add_edge("R", "C",probability=0.8)
        with self.assertRaises(ValueError):
            ScenarioTreeModelFromNetworkX(G)

    def test_bad_probability2(self):
        G = networkx.DiGraph()
        G.add_node("R")
        G.add_node("C1")
        G.add_edge("R", "C1", probability=0.8)
        G.add_node("C2")
        G.add_edge("R", "C2", probability=0.1)
        with self.assertRaises(ValueError):
            ScenarioTreeModelFromNetworkX(G)

    def test_bad_custom_stage_names1(self):
        G = networkx.DiGraph()
        G.add_node("R",)
        G.add_node("C1")
        G.add_edge("R", "C1", probability=1.0)
        with self.assertRaises(ValueError):
            ScenarioTreeModelFromNetworkX(
                G, stage_names=["Stage1"])

    def test_bad_custom_stage_names2(self):
        G = networkx.DiGraph()
        G.add_node("R")
        G.add_node("C1")
        G.add_edge("R", "C1", probability=1.0)
        with self.assertRaises(ValueError):
            ScenarioTreeModelFromNetworkX(
                G, stage_names=["Stage1","Stage1"])

    def test_two_stage(self):
        G = networkx.DiGraph()
        G.add_node("Root")
        G.add_node("Child1")
        G.add_edge("Root", "Child1", probability=0.8)
        G.add_node("Child2")
        G.add_edge("Root", "Child2", probability=0.2)
        model = ScenarioTreeModelFromNetworkX(G)
        self.assertEqual(
            sorted(list(model.Stages)),
            sorted(["Stage1", "Stage2"]))
        self.assertEqual(
            sorted(list(model.Nodes)),
            sorted(["Root", "Child1", "Child2"]))
        self.assertEqual(
            sorted(list(model.Children["Root"])),
            sorted(["Child1", "Child2"]))
        self.assertEqual(
            sorted(list(model.Children["Child1"])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Children["Child2"])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Scenarios)),
            sorted(["Child1", "Child2"]))
        self.assertEqual(model.ConditionalProbability["Root"], 1.0)
        self.assertEqual(model.ConditionalProbability["Child1"], 0.8)
        self.assertEqual(model.ConditionalProbability["Child2"], 0.2)
        model.StageCost["Stage1"] = "c1"
        model.StageCost["Stage2"] = "c2"
        model.StageVariables["Stage1"].add("x")
        ScenarioTree(scenariotreeinstance=model)

    def test_two_stage_custom_names(self):
        G = networkx.DiGraph()
        G.add_node("R", label="Root")
        G.add_node("C1", label="Child1", scenario="S1")
        G.add_edge("R", "C1", weight=0.8)
        G.add_node("C2", label="Child2", scenario="S2")
        G.add_edge("R", "C2", weight=0.2)
        model = ScenarioTreeModelFromNetworkX(
            G,
            edge_probability_attribute="weight",
            node_name_attribute="label",
            stage_names=["T1","T2"],
            scenario_name_attribute="scenario")
        self.assertEqual(
            sorted(list(model.Stages)),
            sorted(["T1", "T2"]))
        self.assertEqual(
            sorted(list(model.Nodes)),
            sorted(["Root", "Child1", "Child2"]))
        self.assertEqual(
            sorted(list(model.Children["Root"])),
            sorted(["Child1", "Child2"]))
        self.assertEqual(
            sorted(list(model.Children["Child1"])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Children["Child2"])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Scenarios)),
            sorted(["S1", "S2"]))
        self.assertEqual(model.ConditionalProbability["Root"], 1.0)
        self.assertEqual(model.ConditionalProbability["Child1"], 0.8)
        self.assertEqual(model.ConditionalProbability["Child2"], 0.2)
        model.StageCost["T1"] = "c1"
        model.StageCost["T2"] = "c2"
        model.StageVariables["T1"].add("x")
        ScenarioTree(scenariotreeinstance=model)

    def test_multi_stage(self):
        G = networkx.balanced_tree(3,2,networkx.DiGraph())
        model = ScenarioTreeModelFromNetworkX(
            G,
            edge_probability_attribute=None)
        self.assertEqual(
            sorted(list(model.Stages)),
            sorted(["Stage1", "Stage2", "Stage3"]))
        self.assertEqual(
            sorted(list(model.Nodes)),
            sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
        self.assertEqual(
            sorted(list(model.Children[0])),
            sorted([1,2,3]))
        self.assertEqual(
            sorted(list(model.Children[1])),
            sorted([4,5,6]))
        self.assertEqual(
            sorted(list(model.Children[2])),
            sorted([7,8,9]))
        self.assertEqual(
            sorted(list(model.Children[3])),
            sorted([10,11,12]))
        self.assertEqual(
            sorted(list(model.Children[4])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Children[5])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Children[6])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Children[7])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Children[8])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Children[9])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Children[10])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Children[11])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Children[12])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Scenarios)),
            sorted([4, 5, 6, 7, 8, 9, 10, 11, 12]))
        self.assertEqual(model.ConditionalProbability[0], 1.0)
        self.assertAlmostEqual(model.ConditionalProbability[1], 1.0/3)
        self.assertAlmostEqual(model.ConditionalProbability[2], 1.0/3)
        self.assertAlmostEqual(model.ConditionalProbability[3], 1.0/3)
        self.assertAlmostEqual(model.ConditionalProbability[4], 1.0/3)
        self.assertAlmostEqual(model.ConditionalProbability[5], 1.0/3)
        self.assertAlmostEqual(model.ConditionalProbability[6], 1.0/3)
        self.assertAlmostEqual(model.ConditionalProbability[7], 1.0/3)
        self.assertAlmostEqual(model.ConditionalProbability[8], 1.0/3)
        self.assertAlmostEqual(model.ConditionalProbability[9], 1.0/3)
        self.assertAlmostEqual(model.ConditionalProbability[10], 1.0/3)
        self.assertAlmostEqual(model.ConditionalProbability[11], 1.0/3)
        self.assertAlmostEqual(model.ConditionalProbability[12], 1.0/3)
        model.StageCost["Stage1"] = "c1"
        model.StageCost["Stage2"] = "c2"
        model.StageCost["Stage3"] = "c3"
        model.StageVariables["Stage1"].add("x")
        model.StageVariables["Stage2"].add("y")
        model.StageVariables["Stage3"].add("y")
        ScenarioTree(scenariotreeinstance=model)


    def test_unbalanced(self):
        G = networkx.DiGraph()
        G.add_node("R")
        G.add_node("0")
        G.add_node("1")
        G.add_edge("R", "0")
        G.add_edge("R", "1")
        G.add_node("00")
        G.add_node("01")
        G.add_edge("0", "00")
        G.add_edge("0", "01")
        model = ScenarioTreeModelFromNetworkX(
            G,
            edge_probability_attribute=None)
        self.assertEqual(
            sorted(list(model.Stages)),
            sorted(["Stage1", "Stage2", "Stage3"]))
        self.assertEqual(
            sorted(list(model.Nodes)),
            sorted(["R","0","1","00","01"]))
        self.assertEqual(
            sorted(list(model.Children["R"])),
            sorted(["0", "1"]))
        self.assertEqual(
            sorted(list(model.Children["0"])),
            sorted(["00","01"]))
        self.assertEqual(
            sorted(list(model.Children["1"])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Children["00"])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Children["01"])),
            sorted([]))
        self.assertEqual(
            sorted(list(model.Scenarios)),
            sorted(["00", "01", "1"]))
        self.assertEqual(model.ConditionalProbability["R"], 1.0)
        self.assertEqual(model.ConditionalProbability["0"], 0.5)
        self.assertEqual(model.ConditionalProbability["1"], 0.5)
        self.assertEqual(model.ConditionalProbability["00"], 0.5)
        self.assertEqual(model.ConditionalProbability["01"], 0.5)
        model.StageCost["Stage1"] = "c1"
        model.StageCost["Stage2"] = "c2"
        model.StageCost["Stage3"] = "c3"
        model.StageVariables["Stage1"].add("x")
        model.StageVariables["Stage2"].add("x")
        ScenarioTree(scenariotreeinstance=model)

TestScenarioTree = unittest.category('smoke','nightly','expensive')(TestScenarioTree)
TestScenarioTreeFromNetworkX = unittest.category('smoke','nightly','expensive')(TestScenarioTreeFromNetworkX)

if __name__ == "__main__":
    unittest.main()
