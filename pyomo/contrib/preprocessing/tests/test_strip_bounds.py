"""Tests stripping of variable bounds."""
import pyutilib.th as unittest

from pyomo.environ import (Binary, ConcreteModel, Integers, NonNegativeReals,
                           PositiveReals, Reals, TransformationFactory, Var)


class TestStripBounds(unittest.TestCase):
    """Tests stripping of bounds."""

    def shortDescription(self):
        """Suppress one-line description at high verbosity."""
        return None

    def test_strip_bounds_maps_exist(self):
        """Tests if component maps for reversion already exist."""
        m = ConcreteModel()
        m.v0 = Var(bounds=(2, 4))
        m.v1 = Var(domain=NonNegativeReals)
        m.v2 = Var(domain=PositiveReals)
        m.v3 = Var(bounds=(-1, 1))
        m.v4 = Var(domain=Binary)
        m.v5 = Var(domain=Integers, bounds=(15, 16))

        xfrm = TransformationFactory('contrib.strip_var_bounds')
        xfrm.apply_to(m, reversible=True)
        # At this point, component maps for reversion already exist.
        with self.assertRaises(RuntimeError):
            xfrm.apply_to(m, reversible=True)

    def test_strip_bounds(self):
        """Test bound stripping and restoration."""
        m = ConcreteModel()
        m.v0 = Var(bounds=(2, 4))
        m.v1 = Var(domain=NonNegativeReals)
        m.v2 = Var(domain=PositiveReals)
        m.v3 = Var(bounds=(-1, 1))
        m.v4 = Var(domain=Binary)
        m.v5 = Var(domain=Integers, bounds=(15, 16))

        xfrm = TransformationFactory('contrib.strip_var_bounds')
        xfrm.apply_to(m, reversible=True)
        self.assertEqual(m.v0.bounds, (None, None))
        self.assertEqual(m.v1.bounds, (None, None))
        self.assertEqual(m.v2.bounds, (None, None))
        self.assertEqual(m.v3.bounds, (None, None))
        self.assertEqual(m.v4.bounds, (None, None))
        self.assertEqual(m.v5.bounds, (None, None))
        self.assertEqual(m.v0.domain, Reals)
        self.assertEqual(m.v1.domain, Reals)
        self.assertEqual(m.v2.domain, Reals)
        self.assertEqual(m.v3.domain, Reals)
        self.assertEqual(m.v4.domain, Reals)
        self.assertEqual(m.v5.domain, Reals)

        xfrm.revert(m)
        self.assertEqual(m.v0.bounds, (2, 4))
        self.assertEqual(m.v1.bounds, (0, None))
        self.assertEqual(m.v2.bounds, (0, None))
        self.assertEqual(m.v3.bounds, (-1, 1))
        self.assertEqual(m.v4.bounds, (0, 1))
        self.assertEqual(m.v5.bounds, (15, 16))
        self.assertEqual(m.v0.domain, Reals)
        self.assertEqual(m.v1.domain, NonNegativeReals)
        self.assertEqual(m.v2.domain, PositiveReals)
        self.assertEqual(m.v3.domain, Reals)
        self.assertEqual(m.v4.domain, Binary)
        self.assertEqual(m.v5.domain, Integers)

    def test_no_strip_domain(self):
        """Test bounds stripping without domain change."""
        m = ConcreteModel()
        m.v0 = Var(bounds=(2, 4))
        m.v1 = Var(domain=NonNegativeReals)
        m.v2 = Var(domain=PositiveReals)
        m.v3 = Var(bounds=(-1, 1))
        m.v4 = Var(domain=Binary)
        m.v5 = Var(domain=Integers, bounds=(15, 16))

        xfrm = TransformationFactory('contrib.strip_var_bounds')
        xfrm.apply_to(m, strip_domains=False, reversible=True)
        self.assertEqual(m.v0.bounds, (None, None))
        self.assertEqual(m.v1.bounds, (0, None))
        self.assertEqual(m.v2.bounds, (0, None))
        self.assertEqual(m.v3.bounds, (None, None))
        self.assertEqual(m.v4.bounds, (0, 1))
        self.assertEqual(m.v5.bounds, (None, None))
        self.assertEqual(m.v0.domain, Reals)
        self.assertEqual(m.v1.domain, NonNegativeReals)
        self.assertEqual(m.v2.domain, PositiveReals)
        self.assertEqual(m.v3.domain, Reals)
        self.assertEqual(m.v4.domain, Binary)
        self.assertEqual(m.v5.domain, Integers)

        xfrm.revert(m)
        self.assertEqual(m.v0.bounds, (2, 4))
        self.assertEqual(m.v1.bounds, (0, None))
        self.assertEqual(m.v2.bounds, (0, None))
        self.assertEqual(m.v3.bounds, (-1, 1))
        self.assertEqual(m.v4.bounds, (0, 1))
        self.assertEqual(m.v5.bounds, (15, 16))
        self.assertEqual(m.v0.domain, Reals)
        self.assertEqual(m.v1.domain, NonNegativeReals)
        self.assertEqual(m.v2.domain, PositiveReals)
        self.assertEqual(m.v3.domain, Reals)
        self.assertEqual(m.v4.domain, Binary)
        self.assertEqual(m.v5.domain, Integers)


if __name__ == '__main__':
    unittest.main()
