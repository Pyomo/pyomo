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

from pyomo.common.collections import ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
import pyomo.common.unittest as unittest
from pyomo.core import Block, ConcreteModel, Constraint, Objective, Var
from pyomo.util.config_domains import ComponentDataSet


def ComponentSetConfig():
    CONFIG = ConfigDict()
    CONFIG.declare(
        'var_set',
        ConfigValue(default=None, domain=ComponentDataSet(Var), doc="VarDataSet"),
    )
    CONFIG.declare(
        'var_and_constraint_set',
        ConfigValue(
            default=None,
            domain=ComponentDataSet(ctype=(Var, Constraint)),
            doc="VarAndConstraintSet",
        ),
    )
    return CONFIG


def a_model():
    m = ConcreteModel()
    m.x = Var()
    m.y = Var([1, 2])
    m.c = Constraint(expr=m.x + m.y[1] <= 3)
    m.c2 = Constraint(expr=m.y[2] >= 7)
    m.obj = Objective(expr=m.x + m.y[1] + m.y[2])
    m.b = Block()

    return m


class TestComponentDataSetDomain(unittest.TestCase):
    def test_var_set(self):
        m = a_model()
        config = ComponentSetConfig()
        self.assertIsNone(config.var_set)
        config.var_set = ComponentSet([m.x, m.y])
        self.assertIsInstance(config.var_set, ComponentSet)
        self.assertEqual(len(config.var_set), 3)
        for v in [m.x, m.y[1], m.y[2]]:
            self.assertIn(v, config.var_set)

        with self.assertRaisesRegex(
            ValueError,
            ".*Expected component or iterable of one "
            "of the following ctypes: Var.\n\t"
            "Received <class 'pyomo.core.base.constraint.ScalarConstraint'>",
        ):
            config.var_set = ComponentSet([m.y, m.c])

    def test_var_and_constraint_set(self):
        m = a_model()
        config = ComponentSetConfig()
        self.assertIsNone(config.var_and_constraint_set)
        config.var_and_constraint_set = ComponentSet([m.x, m.c])
        self.assertIsInstance(config.var_and_constraint_set, ComponentSet)
        self.assertEqual(len(config.var_and_constraint_set), 2)
        for v in [m.x, m.c]:
            self.assertIn(v, config.var_and_constraint_set)

        with self.assertRaisesRegex(
            ValueError,
            ".*Expected component or iterable of one "
            "of the following ctypes: Constraint, Var.\n\t"
            "Received <class 'pyomo.core.base.block.ScalarBlock'>",
        ):
            config.var_and_constraint_set = ComponentSet([m.y, m.c, m.b])

        with self.assertRaisesRegex(
            ValueError,
            ".*Expected component or iterable of one "
            "of the following ctypes: Constraint, Var.\n\t"
            "Received <class 'int'>",
        ):
            config.var_and_constraint_set = ComponentSet([3, m.y, m.c])

    def test_domain_name(self):
        config = ComponentSetConfig()
        self.assertEqual(config.get("var_set").domain_name(), "ComponentDataSet(Var)")
        self.assertEqual(
            config.get("var_and_constraint_set").domain_name(),
            "ComponentDataSet([Constraint, Var])",
        )
