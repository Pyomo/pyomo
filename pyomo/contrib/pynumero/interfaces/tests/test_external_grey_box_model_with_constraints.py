# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
#
#  Additional contributions Copyright (c) 2026 OLI Systems, Inc.
#  ___________________________________________________________________________________


import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet

from pyomo.contrib.pynumero.dependencies import (
    numpy_available,
    scipy_available,
)

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.asl import AmplInterface

if not AmplInterface.available():
    raise unittest.SkipTest("ASL interface is not available")

from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface


class TestExternalGreyBoxModelWithIncidenceAnalysis(unittest.TestCase):
    """Tests for integration of ExternalGreyBoxBlock with incidence analysis"""

    def build_model(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualitiesTwoOutputsWithHessian()
        m.egb.set_external_model(external_model)

        return m

    def build_model_with_pyomo_components(self):
        m = self.build_model()

        # Add Vars and linking constraints to m
        m.Pin = pyo.Var()
        m.c = pyo.Var()
        m.F = pyo.Var()
        m.P1 = pyo.Var()
        m.P3 = pyo.Var()
        m.P2 = pyo.Var()
        m.Pout = pyo.Var()

        m.link_Pin = pyo.Constraint(expr=m.Pin == m.egb.inputs["Pin"])
        m.link_c = pyo.Constraint(expr=m.c == m.egb.inputs["c"])
        m.link_F = pyo.Constraint(expr=m.F == m.egb.inputs["F"])
        m.link_P1 = pyo.Constraint(expr=m.P1 == m.egb.inputs["P1"])
        m.link_P3 = pyo.Constraint(expr=m.P3 == m.egb.inputs["P3"])
        m.link_P2 = pyo.Constraint(expr=m.P2 == m.egb.outputs["P2"])
        m.link_Pout = pyo.Constraint(expr=m.Pout == m.egb.outputs["Pout"])

        return m

    def test_grey_box_only(self):
        """
        Test that the incidence analysis correctly determines the DM partition for
        a grey box model with two equality constraints and two outputs
        """
        m = self.build_model()

        # Check that the identify_variables method on the implicit constraint body returns the correct variables
        # Implicit constraint: 'pdrop1'
        body_obj1 = m.egb.eq_constraints["pdrop1"].body
        incident_vars1 = body_obj1.identify_variables()

        incident_var_set = ComponentSet(
            [
                m.egb.inputs["Pin"],
                m.egb.inputs["c"],
                m.egb.inputs["F"],
                m.egb.inputs["P1"],
            ]
        )
        self.assertEqual(ComponentSet(incident_vars1), incident_var_set)

        # Implicit constraint: 'pdrop3'
        body_obj1 = m.egb.eq_constraints["pdrop3"].body
        incident_vars1 = body_obj1.identify_variables()
        incident_var_set = ComponentSet(
            [
                m.egb.inputs["c"],
                m.egb.inputs["F"],
                m.egb.inputs["P1"],
                m.egb.inputs["P3"],
            ]
        )
        self.assertEqual(ComponentSet(incident_vars1), incident_var_set)

        # Implicit constraint: 'P2_constraint'
        body_obj1 = m.egb.output_constraints["P2"].body
        incident_vars1 = body_obj1.identify_variables()
        incident_var_set = ComponentSet(
            [
                m.egb.inputs["c"],
                m.egb.inputs["F"],
                m.egb.inputs["P1"],
                m.egb.outputs["P2"],
            ]
        )
        self.assertEqual(ComponentSet(incident_vars1), incident_var_set)

        # Implicit constraint: 'Pout_constraint'
        body_obj1 = m.egb.output_constraints["Pout"].body
        incident_vars1 = body_obj1.identify_variables()
        incident_var_set = ComponentSet(
            [
                m.egb.inputs["Pin"],
                m.egb.inputs["c"],
                m.egb.inputs["F"],
                m.egb.outputs["Pout"],
            ]
        )
        self.assertEqual(ComponentSet(incident_vars1), incident_var_set)

        # Check Dulmage-Mendelsohn partitioning of the incidence graph
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        var_dm_partition, con_dm_partition = igraph.dulmage_mendelsohn()

        # In this case, as we have not fixed any variables, we expect the system to be under-constrained.
        # All variables will be in the under-constrained or unmatched sets
        # All constraints will be in the under-constrained set
        self.assertEqual(var_dm_partition.overconstrained, [])
        self.assertEqual(var_dm_partition.square, [])
        self.assertEqual(con_dm_partition.unmatched, [])
        self.assertEqual(con_dm_partition.overconstrained, [])
        self.assertEqual(con_dm_partition.square, [])

        self.assertEqual(len(var_dm_partition.underconstrained), 4)
        self.assertEqual(len(var_dm_partition.unmatched), 3)

        for v in var_dm_partition.underconstrained:
            # output variables should be in the under-constrained set
            # The other two variables should be drawn from the inputs, but we cannot guarantee which ones
            self.assertIn(
                v.name,
                [
                    "egb.inputs[Pin]",
                    "egb.inputs[c]",
                    "egb.inputs[F]",
                    "egb.inputs[P1]",
                    "egb.inputs[P3]",
                    "egb.outputs[P2]",
                    "egb.outputs[Pout]",
                ],
            )

        for v in var_dm_partition.unmatched:
            # Unmatched set will have the remaining 3 input variables, but again we cannot guarantee which ones
            # We will instead check that the name is one of the inputs and that it is not in the under-constrained set
            self.assertNotIn(
                v.name, [u.name for u in var_dm_partition.underconstrained]
            )
            self.assertIn(
                v.name,
                [
                    "egb.inputs[Pin]",
                    "egb.inputs[c]",
                    "egb.inputs[F]",
                    "egb.inputs[P1]",
                    "egb.inputs[P3]",
                ],
            )

        self.assertEqual(len(con_dm_partition.underconstrained), 4)
        con_names = [c.name for c in con_dm_partition.underconstrained]
        for c in con_names:
            self.assertIn(
                c,
                [
                    "egb.eq_constraints[pdrop1]",
                    "egb.eq_constraints[pdrop3]",
                    "egb.output_constraints[P2]",
                    "egb.output_constraints[Pout]",
                ],
            )

    def test_grey_box_w_pyomo_components(self):
        """
        Test that the incidence analysis correctly determines the DM partition for
        a model containing both grey box and other components
        """
        m = self.build_model_with_pyomo_components()

        # Check Dulmage-Mendelsohn partitioning of the incidence graph
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        var_dm_partition, con_dm_partition = igraph.dulmage_mendelsohn()

        # In this case, as we have not fixed any variables, we expect the system to be under-constrained.
        # All variables will be in the under-constrained or unmatched sets
        # All constraints will be in the under-constrained set
        self.assertEqual(var_dm_partition.overconstrained, [])
        self.assertEqual(var_dm_partition.square, [])
        self.assertEqual(con_dm_partition.unmatched, [])
        self.assertEqual(con_dm_partition.overconstrained, [])
        self.assertEqual(con_dm_partition.square, [])

        self.assertEqual(len(var_dm_partition.underconstrained), 11)
        self.assertEqual(len(var_dm_partition.unmatched), 3)
        var_names = [
            v.name
            for v in var_dm_partition.underconstrained + var_dm_partition.unmatched
        ]
        for v in var_names:
            self.assertIn(
                v,
                [
                    "egb.inputs[Pin]",
                    "egb.inputs[c]",
                    "egb.inputs[F]",
                    "egb.inputs[P1]",
                    "egb.inputs[P3]",
                    "egb.outputs[P2]",
                    "egb.outputs[Pout]",
                    "Pin",
                    "c",
                    "F",
                    "P1",
                    "P3",
                    "P2",
                    "Pout",
                ],
            )

        self.assertEqual(len(con_dm_partition.underconstrained), 11)
        con_names = [c.name for c in con_dm_partition.underconstrained]
        for c in con_names:
            self.assertIn(
                c,
                [
                    "egb.eq_constraints[pdrop1]",
                    "egb.eq_constraints[pdrop3]",
                    "egb.output_constraints[P2]",
                    "egb.output_constraints[Pout]",
                    "link_Pin",
                    "link_c",
                    "link_F",
                    "link_P1",
                    "link_P3",
                    "link_P2",
                    "link_Pout",
                ],
            )

    def test_grey_box_w_pyomo_components_square(self):
        """
        Test that the incidence analysis correctly determines the DM partition for
        a model containing both grey box and other components
        """
        m = self.build_model_with_pyomo_components()

        # Fix 3 inputs
        # Note that we have 2 implicit constraints that cross-link inputs
        m.Pin.fix(1)
        m.c.fix(1)
        m.F.fix(1)

        # Check Dulmage-Mendelsohn partitioning of the incidence graph
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        var_dm_partition, con_dm_partition = igraph.dulmage_mendelsohn()

        # In this case, as we have fixed all input variables, we expect the system to be square.
        # All variables and constraints will be in the square sets
        self.assertEqual(var_dm_partition.unmatched, [])
        self.assertEqual(var_dm_partition.overconstrained, [])
        self.assertEqual(var_dm_partition.underconstrained, [])
        self.assertEqual(con_dm_partition.unmatched, [])
        self.assertEqual(con_dm_partition.overconstrained, [])
        self.assertEqual(con_dm_partition.underconstrained, [])

        self.assertEqual(len(var_dm_partition.square), 11)
        var_names = [v.name for v in var_dm_partition.square]
        for v in var_names:
            self.assertIn(
                v,
                [
                    "egb.inputs[Pin]",
                    "egb.inputs[c]",
                    "egb.inputs[F]",
                    "egb.inputs[P1]",
                    "egb.inputs[P3]",
                    "egb.outputs[P2]",
                    "egb.outputs[Pout]",
                    # These three re fixed, so do not appear by default
                    # 'Pin',
                    # 'c',
                    # 'F',
                    "P1",
                    "P3",
                    "P2",
                    "Pout",
                ],
            )

        self.assertEqual(len(con_dm_partition.square), 11)
        con_names = [c.name for c in con_dm_partition.square]
        for c in con_names:
            self.assertIn(
                c,
                [
                    "egb.eq_constraints[pdrop1]",
                    "egb.eq_constraints[pdrop3]",
                    "egb.output_constraints[P2]",
                    "egb.output_constraints[Pout]",
                    "link_Pin",
                    "link_c",
                    "link_F",
                    "link_P1",
                    "link_P3",
                    "link_P2",
                    "link_Pout",
                ],
            )


if __name__ == "__main__":
    unittest.main()
