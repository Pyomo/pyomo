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

import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models


class TestExternalGreyBoxAsNLP(unittest.TestCase):
    def test_pressure_drop_single_output(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(
            ex_models.PressureDropSingleOutput(),
            build_implicit_constraint_objects=True,
        )
    
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        var_dm_partition, con_dm_partition = igraph.dulmage_mendelsohn()

        uc_var = var_dm_partition.unmatched + var_dm_partition.underconstrained
        uc_con = con_dm_partition.underconstrained
        oc_var = var_dm_partition.overconstrained
        oc_con = con_dm_partition.overconstrained + con_dm_partition.unmatched

        assert len(uc_var) == 4
        for i in uc_var:
            assert i.name in [
                "egb.inputs[Pin]",
                "egb.inputs[c]",
                "egb.inputs[F]",
                "egb.outputs[Pout]"
            ]
        assert len(uc_con) == 1
        assert uc_con[0].name == "egb.Pout_constraint"
        assert len(oc_var) == 0
        assert len(oc_con) == 0

        max_matching = igraph.maximum_matching()
        assert len(max_matching) == 1
        for k,v in max_matching.items():
            assert k.name == "egb.Pout_constraint"
            assert v.name == "egb.outputs[Pout]"

        con_vars, con_cons = igraph.get_connected_components()
        assert len(con_vars) == 1
        assert len(con_cons) == 1
        assert len(con_vars[0]) == 4
        for j in con_vars[0]:
            assert j.name in [
                "egb.inputs[Pin]",
                "egb.inputs[c]",
                "egb.inputs[F]",
                "egb.outputs[Pout]"
            ]
        assert len(con_cons[0]) == 1
        for j in con_cons[0]:
            assert j.name in [
                "egb.Pout_constraint"
            ]

        # Add constraints to make model square, then rebuild graph to test block triangularization
        m.con1 = pyo.Constraint(expr=m.egb.inputs["Pin"] == 1)
        m.con2 = pyo.Constraint(expr=m.egb.inputs["c"] == 1)
        m.con3 = pyo.Constraint(expr=m.egb.inputs["F"] == 1)
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        bt_vars, bt_cons = igraph.block_triangularize()

        # Expect 4 decomposable sub-sets, one for each linking constraint and one for the grey box
        assert len(bt_vars) == 4
        assert len(bt_cons) == 4

        matchings = {
            "egb.inputs[Pin]": "con1",
            "egb.inputs[c]": "con2",
            "egb.inputs[F]": "con3",
            "egb.outputs[Pout]": "egb.Pout_constraint",
        }

        for i in range(len(bt_vars)):
            assert len(bt_vars[i]) == 1
            assert len(bt_cons[i]) == 1
            
            match_var = bt_vars[i][0].name
            match_con = bt_cons[i][0].name

            assert match_con == matchings[match_var]

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

        assert len(uc_var) == 7
        for i in uc_var:
            assert i.name in [
                "egb.inputs[F]",
                "egb.inputs[P1]",
                "egb.inputs[P3]",
                "egb.inputs[Pin]",
                "egb.inputs[c]",
                "egb.outputs[P2]",
                "egb.outputs[Pout]"
            ]
        assert len(uc_con) == 4
        for i in uc_con:
            assert i.name in [
                "egb.Pout_constraint",
                "egb.P2_constraint",
                "egb.pdrop1",
                "egb.pdrop3",
            ]
        assert len(oc_var) == 0
        assert len(oc_con) == 0

        max_matching = igraph.maximum_matching()
        assert len(max_matching) == 4
        expected_matches = {
            "egb.pdrop1": "egb.inputs[Pin]",
            "egb.pdrop3": "egb.inputs[c]",
            "egb.P2_constraint": "egb.outputs[P2]",
            "egb.Pout_constraint": "egb.outputs[Pout]"
        }
        for k, v in max_matching.items():
            assert v.name == expected_matches[k.name]

        con_vars, con_cons = igraph.get_connected_components()
        assert len(con_vars) == 1
        assert len(con_cons) == 1

        assert len(con_vars[0]) == 7
        for j in con_vars[0]:
            assert j.name in [
                "egb.inputs[F]",
                "egb.inputs[P1]",
                "egb.inputs[P3]",
                "egb.inputs[Pin]",
                "egb.inputs[c]",
                "egb.outputs[P2]",
                "egb.outputs[Pout]"
            ]
        assert len(con_cons[0]) == 4
        for j in con_cons[0]:
            assert j.name in [
                "egb.Pout_constraint",
                "egb.P2_constraint",
                "egb.pdrop1",
                "egb.pdrop3",
            ]

        # Add constraints to make model square, then rebuild graph to test block triangularization
        m.con1 = pyo.Constraint(expr=m.egb.inputs["F"] == 1)
        m.con2 = pyo.Constraint(expr=m.egb.inputs["Pin"] == 1)
        m.con3 = pyo.Constraint(expr=m.egb.inputs["c"] == 1)
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        bt_vars, bt_cons = igraph.block_triangularize()

        for i, v in enumerate(bt_vars):
            print(f"\nBlock {i}\n")
            for j in v:
                print(j.name)
            for j in bt_cons[i]:
                print(j.name)

        # Get 6 decomposable sub-sets
        # 3 linking constraints give 3 sub-sets
        # Grey box gets broken into 3 parts for some reason
        assert len(bt_vars) == 6
        assert len(bt_cons) == 6

        for i in range(len(bt_vars)):
            assert len(bt_vars[i]) == len(bt_cons[i])
        
        # Block 0
        assert len(bt_vars[0]) == 1
        assert len(bt_cons[0]) == 1
        assert bt_vars[0][0].name == "egb.inputs[F]"
        assert bt_cons[0][0].name == "con1"

        # Block 1
        assert len(bt_vars[1]) == 1
        assert len(bt_cons[1]) == 1
        assert bt_vars[1][0].name == "egb.inputs[Pin]"
        assert bt_cons[1][0].name == "con2"

        # Block 2
        assert len(bt_vars[2]) == 1
        assert len(bt_cons[2]) == 1
        assert bt_vars[2][0].name == "egb.inputs[c]"
        assert bt_cons[2][0].name == "con3"

        # Block 3
        assert len(bt_vars[3]) == 2
        assert len(bt_cons[3]) == 2

        for i in bt_vars[3]:
            assert i.name in [
                "egb.inputs[P1]",
                "egb.inputs[P3]",
            ]
        for i in bt_cons[3]:
            assert i.name in [
                "egb.pdrop1",
                "egb.pdrop3",
            ]
        
        # Block 4
        assert len(bt_vars[4]) == 1
        assert len(bt_cons[4]) == 1
        assert bt_vars[4][0].name == "egb.outputs[P2]"
        assert bt_cons[4][0].name == "egb.P2_constraint"

        # Block 5
        assert len(bt_vars[5]) == 1
        assert len(bt_cons[5]) == 1
        assert bt_vars[5][0].name == "egb.outputs[Pout]"
        assert bt_cons[5][0].name == "egb.Pout_constraint"
