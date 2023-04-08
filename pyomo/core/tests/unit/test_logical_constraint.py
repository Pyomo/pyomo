import pyomo.common.unittest as unittest

from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.environ import (
    AbstractModel,
    BooleanVar,
    ConcreteModel,
    LogicalConstraint,
    TransformationFactory,
    Constraint,
)
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunction


class TestLogicalConstraintCreation(unittest.TestCase):
    def create_model(self, abstract=False):
        if abstract is True:
            model = AbstractModel()
        else:
            model = ConcreteModel()
        model.x = BooleanVar()
        model.y = BooleanVar()
        model.z = BooleanVar()
        return model

    def test_construct(self):
        model = self.create_model()

        def rule(model):
            return model.x

        model.p = LogicalConstraint(rule=rule)

        self.assertIs(model.p.body, model.x)

    def check_lor_on_disjunct(self, model, disjunct, x1, x2):
        x1 = x1.get_associated_binary()
        x2 = x2.get_associated_binary()
        disj0 = disjunct.logic_to_linear
        self.assertEqual(len(disj0.component_map(Constraint)), 1)
        lor = disj0.transformed_constraints[1]
        self.assertEqual(lor.lower, 1)
        self.assertIsNone(lor.upper)
        repn = generate_standard_repn(lor.body)
        self.assertEqual(repn.constant, 0)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertIs(repn.linear_vars[0], x1)
        self.assertIs(repn.linear_vars[1], x2)
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertEqual(repn.linear_coefs[1], 1)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_statement_in_Disjunct_with_logical_to_linear(self):
        # This is an old test that originally tested that GDP's
        # BigM/Hull correctly handled Disjuncts with LogicalConstraints
        # (implicitly calling logical_to_linear to leave the transformed
        # algebraic constraints on the Disjuncts).  That is no longer
        # the default behavior.  However, we will preserve this (with an
        # explicit call to logical_to_linear) for posterity
        model = self.create_model()
        model.disj = Disjunction(expr=[[model.x.lor(model.y)], [model.y.lor(model.z)]])

        TransformationFactory('core.logical_to_linear').apply_to(
            model, targets=model.disj.disjuncts
        )

        bigmed = TransformationFactory('gdp.bigm').create_using(model)
        # check that the algebraic versions are living on the Disjuncts
        self.check_lor_on_disjunct(bigmed, bigmed.disj.disjuncts[0], bigmed.x, bigmed.y)
        self.check_lor_on_disjunct(bigmed, bigmed.disj.disjuncts[1], bigmed.y, bigmed.z)

        TransformationFactory('gdp.hull').apply_to(model)
        self.check_lor_on_disjunct(model, model.disj.disjuncts[0], model.x, model.y)
        self.check_lor_on_disjunct(model, model.disj.disjuncts[1], model.y, model.z)

    # TODO look to test_con.py for inspiration


if __name__ == "__main__":
    unittest.main()
