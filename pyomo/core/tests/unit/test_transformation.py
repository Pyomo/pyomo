#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

from pyomo.core.base.plugin import UnknownTransformation
from pyomo.core.plugins.transform.discrete_vars import RelaxIntegerVars
from pyomo.environ import TransformationFactory, ConcreteModel

class TestTransformationFactory(unittest.TestCase):
    def test_existing_solver(self):
        xfrm = TransformationFactory('core.relax_integer_vars')
        self.assertIs(type(xfrm), RelaxIntegerVars)

    def test_unknown_solver(self):
        m = ConcreteModel()
        xfrm = TransformationFactory('core.__bad_transform')
        self.assertIs(type(xfrm), UnknownTransformation)
        with self.assertRaisesRegex(
                RuntimeError,
                '(?s)Attempting to use an unavailable transformation'
                '.*unable to create "core.__bad_transform"'
                '.*by calling method "apply".*{}'):
            xfrm.apply(m)

        with self.assertRaisesRegex(
                RuntimeError,
                '(?s)Attempting to use an unavailable transformation'
                '.*unable to create "core.__bad_transform"'
                '.*by calling method "apply_to".*{}'):
            xfrm.apply_to(m)

        xfrm = TransformationFactory('core.__bad_transform', mykwd='a val')
        with self.assertRaisesRegex(
                RuntimeError,
                '(?s)Attempting to use an unavailable transformation'
                '.*unable to create "core.__bad_transform"'
                '.*by calling method "create_using"'
                ".*mykwd: 'a val'"):
            xfrm.create_using(m)
