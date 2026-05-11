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
from pyomo.common.collections import ComponentSet

from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models
from pyomo.contrib.pynumero.interfaces.external_grey_box_constraint import (
    ExternalGreyBoxConstraint,
)


class TestExternalGreyBoxIncidence(unittest.TestCase):
    def test_pressure_drop_single_output(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_models.PressureDropSingleOutput())

        igraph = IncidenceGraphInterface(m, include_inequality=False)
        var_dm_partition, con_dm_partition = igraph.dulmage_mendelsohn()

        uc_var = var_dm_partition.unmatched + var_dm_partition.underconstrained
        uc_con = con_dm_partition.underconstrained
        oc_var = var_dm_partition.overconstrained
        oc_con = con_dm_partition.overconstrained + con_dm_partition.unmatched

        egb_var = ComponentSet(m.egb.component_data_objects(pyo.Var))
        self.assertEqual(ComponentSet(uc_var), egb_var)

        uc_cons_set = ComponentSet([m.egb.output_constraints["Pout"]])
        self.assertEqual(ComponentSet(uc_con), uc_cons_set)

        self.assertEqual(ComponentSet(oc_var), ComponentSet([]))
        self.assertEqual(ComponentSet(oc_con), ComponentSet([]))

        max_matching = igraph.maximum_matching()
        self.assertIn(max_matching[m.egb.output_constraints["Pout"]], egb_var)

        cc_vars, cc_cons = igraph.get_connected_components()
        self.assertEqual(ComponentSet(cc_vars[0]), egb_var)
        self.assertEqual(cc_cons[0][0].name, "egb.output_constraints[Pout]")

    def test_pressure_drop_single_output_block_triangularization(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_models.PressureDropSingleOutput())

        # Add constraints to make model square, then rebuild graph to test block triangularization
        m.con1 = pyo.Constraint(expr=m.egb.inputs["Pin"] == 1)
        m.con2 = pyo.Constraint(expr=m.egb.inputs["c"] == 1)
        m.con3 = pyo.Constraint(expr=m.egb.inputs["F"] == 1)
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        bt_vars, bt_cons = igraph.block_triangularize()

        # Expect 4 decomposable sub-sets, one for each linking constraint and one for the grey box
        self.assertEqual(len(bt_vars), 4)
        self.assertEqual(len(bt_cons), 4)

        con_set_0 = [m.con1]
        con_set_1 = [m.con2]
        con_set_2 = [m.con3]
        con_set_3 = [m.egb.output_constraints["Pout"]]
        expected_bt_cons = [con_set_0, con_set_1, con_set_2, con_set_3]

        for var_set in bt_vars:
            self.assertEqual(len(var_set), 1)
            # Need to use variable name here to avoid attempting equality checks between variables
            self.assertIn(
                var_set[0].name,
                [
                    "egb.inputs[Pin]",
                    "egb.inputs[c]",
                    "egb.inputs[F]",
                    "egb.outputs[Pout]",
                ],
            )

        for con_set in bt_cons:
            self.assertEqual(len(con_set), 1)
            self.assertIn(con_set, expected_bt_cons)

        self.assertEqual(
            ComponentSet(igraph.get_adjacent_to(m.egb.output_constraints["Pout"])),
            ComponentSet(m.egb.component_data_objects(pyo.Var)),
        )

    def test_pressure_drop_two_equalities_two_outputs(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_models.PressureDropTwoEqualitiesTwoOutputs())

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
                m.egb.output_constraints["Pout"],
                m.egb.output_constraints["P2"],
                m.egb.eq_constraints["pdrop1"],
                m.egb.eq_constraints["pdrop3"],
            ]
        )
        self.assertEqual(ComponentSet(uc_con), uc_con_set)

        self.assertEqual(ComponentSet(oc_var), ComponentSet([]))
        self.assertEqual(ComponentSet(oc_con), ComponentSet([]))

        max_matching = igraph.maximum_matching()
        egb_var = ComponentSet(m.egb.component_data_objects(pyo.Var))
        egb_cons = ComponentSet(m.egb.component_data_objects(ExternalGreyBoxConstraint))
        self.assertIn(max_matching[m.egb.output_constraints["Pout"]], egb_var)

        cc_vars, cc_cons = igraph.get_connected_components()
        self.assertEqual(ComponentSet(cc_vars[0]), egb_var)
        self.assertEqual(ComponentSet(cc_cons[0]), egb_cons)

    def test_pressure_drop_two_equalities_two_outputs_block_triangularization(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_models.PressureDropTwoEqualitiesTwoOutputs())

        # Add constraints to make model square, then rebuild graph to test block triangularization
        m.con1 = pyo.Constraint(expr=m.egb.inputs["F"] == 1)
        m.con2 = pyo.Constraint(expr=m.egb.inputs["Pin"] == 1)
        m.con3 = pyo.Constraint(expr=m.egb.inputs["c"] == 1)
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        bt_vars, bt_cons = igraph.block_triangularize()

        matching = {
            m.con1: m.egb.inputs["F"],
            m.con2: m.egb.inputs["Pin"],
            m.con3: m.egb.inputs["c"],
            m.egb.eq_constraints["pdrop1"]: m.egb.inputs["P1"],
            m.egb.eq_constraints["pdrop3"]: m.egb.inputs["P3"],
            m.egb.output_constraints["P2"]: m.egb.outputs["P2"],
            m.egb.output_constraints["Pout"]: m.egb.outputs["Pout"],
        }

        seen = ComponentSet()
        for vars, cons in zip(bt_vars, bt_cons):
            self.assertEqual(len(vars), 1)
            self.assertIs(vars[0], matching[cons[0]])
            seen.update(vars)
            # We know that P1 has to come before P2 and P3 in the block triangular form
            if vars[0] is m.egb.outputs["P2"] or vars[0] is m.egb.inputs["P3"]:
                self.assertIn(m.egb.inputs["P1"], seen)

        # We know that these constraints have to be in the first three blocks
        self.assertEqual(
            set(bt_cons[i][0] for i in range(3)), set([m.con1, m.con2, m.con3])
        )
