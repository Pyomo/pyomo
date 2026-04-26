#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
#  Additional contributions Copyright (c) 2026 OLI Systems, Inc.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap, ComponentSet

from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models


class TestExternalGreyBoxIncidence(unittest.TestCase):
    def test_pressure_drop_single_output(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(
            ex_models.PressureDropSingleOutput(), build_implicit_constraint_objects=True
        )

        igraph = IncidenceGraphInterface(m, include_inequality=False)
        var_dm_partition, con_dm_partition = igraph.dulmage_mendelsohn()

        uc_var = var_dm_partition.unmatched + var_dm_partition.underconstrained
        uc_con = con_dm_partition.underconstrained
        oc_var = var_dm_partition.overconstrained
        oc_con = con_dm_partition.overconstrained + con_dm_partition.unmatched

        egb_var = ComponentSet(m.egb.component_data_objects(pyo.Var))
        self.assertEqual(ComponentSet(uc_var), egb_var)

        uc_cons_set = ComponentSet([m.egb.Pout_constraint])
        self.assertEqual(ComponentSet(uc_con), uc_cons_set)

        self.assertEqual(ComponentSet(oc_var), ComponentSet([]))
        self.assertEqual(ComponentSet(oc_con), ComponentSet([]))

        max_matching = igraph.maximum_matching()
        expected = ComponentMap({m.egb.Pout_constraint: m.egb.outputs["Pout"]})
        self.assertEqual(max_matching, expected)

        con_vars, con_cons = igraph.get_connected_components()

        con_vars_set = ComponentSet(
            [
                m.egb.inputs["Pin"],
                m.egb.inputs["c"],
                m.egb.inputs["F"],
                m.egb.outputs["Pout"],
            ]
        )
        con_cons_set = ComponentSet([m.egb.Pout_constraint])
        self.assertEqual(ComponentSet(con_vars[0]), con_vars_set)
        self.assertEqual(ComponentSet(con_cons[0]), con_cons_set)

    def test_pressure_drop_single_output_block_triangularization(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(
            ex_models.PressureDropSingleOutput(), build_implicit_constraint_objects=True
        )

        # Add constraints to make model square, then rebuild graph to test block triangularization
        m.con1 = pyo.Constraint(expr=m.egb.inputs["Pin"] == 1)
        m.con2 = pyo.Constraint(expr=m.egb.inputs["c"] == 1)
        m.con3 = pyo.Constraint(expr=m.egb.inputs["F"] == 1)
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        bt_vars, bt_cons = igraph.block_triangularize()

        # Expect 4 decomposable sub-sets, one for each linking constraint and one for the grey box
        self.assertEqual(len(bt_vars), 4)
        self.assertEqual(len(bt_cons), 4)

        var_set_1 = [m.egb.inputs["Pin"]]
        var_set_2 = [m.egb.inputs["c"]]
        var_set_3 = [m.egb.inputs["F"]]
        var_set_4 = [m.egb.outputs["Pout"]]
        expected_bt_vars = [var_set_1, var_set_2, var_set_3, var_set_4]

        con_set_1 = [m.con1]
        con_set_2 = [m.con2]
        con_set_3 = [m.con3]
        con_set_4 = [m.egb.Pout_constraint]
        expected_bt_cons = [con_set_1, con_set_2, con_set_3, con_set_4]

        self.assertEqual(bt_vars, expected_bt_vars)
        self.assertEqual(bt_cons, expected_bt_cons)

    def test_pressure_drop_two_equalities_two_outputs(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(
            ex_models.PressureDropTwoEqualitiesTwoOutputs(),
            build_implicit_constraint_objects=True,
        )

        igraph = IncidenceGraphInterface(m, include_inequality=False)
        var_dm_partition, con_dm_partition = igraph.dulmage_mendelsohn()

        uc_var = var_dm_partition.unmatched + var_dm_partition.underconstrained
        uc_con = con_dm_partition.underconstrained
        oc_var = var_dm_partition.overconstrained
        oc_con = con_dm_partition.overconstrained + con_dm_partition.unmatched

        uc_var_set = ComponentSet(
            [
                m.egb.inputs["F"],
                m.egb.inputs["P1"],
                m.egb.inputs["P3"],
                m.egb.inputs["Pin"],
                m.egb.inputs["c"],
                m.egb.outputs["P2"],
                m.egb.outputs["Pout"],
            ]
        )
        self.assertEqual(ComponentSet(uc_var), uc_var_set)

        uc_con_set = ComponentSet(
            [
                m.egb.Pout_constraint,
                m.egb.P2_constraint,
                m.egb.pdrop1,
                m.egb.pdrop3,
            ]
        )
        self.assertEqual(ComponentSet(uc_con), uc_con_set)


        self.assertEqual(ComponentSet(oc_var), ComponentSet([]))
        self.assertEqual(ComponentSet(oc_con), ComponentSet([]))

        max_matching = igraph.maximum_matching()

        expected_max_match = ComponentMap(
            {
                m.egb.pdrop1: m.egb.inputs["Pin"],
                m.egb.pdrop3: m.egb.inputs["c"],
                m.egb.P2_constraint: m.egb.outputs["P2"],
                m.egb.Pout_constraint: m.egb.outputs["Pout"],
            }
        )
        self.assertEqual(max_matching, expected_max_match)

        con_vars, con_cons = igraph.get_connected_components()

        expected_con_vars = [
            ComponentSet(
                [
                    m.egb.inputs["F"],
                    m.egb.inputs["P1"],
                    m.egb.inputs["P3"],
                    m.egb.inputs["Pin"],
                    m.egb.inputs["c"],
                    m.egb.outputs["P2"],
                    m.egb.outputs["Pout"],
                ]
            )
        ]
        expected_con_cons = [
            ComponentSet(
                [
                    m.egb.Pout_constraint,
                    m.egb.P2_constraint,
                    m.egb.pdrop1,
                    m.egb.pdrop3,
                ]
            )
        ]

        self.assertEqual([ComponentSet(con_vars[0])], expected_con_vars)
        self.assertEqual([ComponentSet(con_cons[0])], expected_con_cons)

    def test_pressure_drop_two_equalities_two_outputs_block_triangularization(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(
            ex_models.PressureDropTwoEqualitiesTwoOutputs(),
            build_implicit_constraint_objects=True,
        )

        # Add constraints to make model square, then rebuild graph to test block triangularization
        m.con1 = pyo.Constraint(expr=m.egb.inputs["F"] == 1)
        m.con2 = pyo.Constraint(expr=m.egb.inputs["Pin"] == 1)
        m.con3 = pyo.Constraint(expr=m.egb.inputs["c"] == 1)
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        bt_vars, bt_cons = igraph.block_triangularize()


        # Get 7 decomposable sub-sets
        # 3 linking constraints give 3 sub-sets
        # Grey box gets broken into parts based on the matching
        expected_bt_vars = [
            ComponentSet([m.egb.inputs["F"]]),
            ComponentSet([m.egb.inputs["Pin"]]),
            ComponentSet([m.egb.inputs["c"]]),
            ComponentSet([m.egb.inputs["P1"]]),
            ComponentSet([m.egb.inputs["P3"]]),
            ComponentSet([m.egb.outputs["P2"]]),
            ComponentSet([m.egb.outputs["Pout"]]),
        ]
        expected_bt_cons = [
            ComponentSet([m.con1]),
            ComponentSet([m.con2]),
            ComponentSet([m.con3]),
            ComponentSet([m.egb.pdrop1]),
            ComponentSet([m.egb.pdrop3]),
            ComponentSet([m.egb.P2_constraint]),
            ComponentSet([m.egb.Pout_constraint]),
        ]

        self.assertEqual([ComponentSet(var_set) for var_set in bt_vars], expected_bt_vars)
        self.assertEqual([ComponentSet(con_set) for con_set in bt_cons], expected_bt_cons)
