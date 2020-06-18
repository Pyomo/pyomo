import pyutilib.th as unittest

from pyomo.environ import AbstractModel, BooleanVar, ConcreteModel, LogicalConstraint, TransformationFactory
from pyomo.gdp import Disjunction, GDP_Error


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

    def test_statement_in_Disjunct(self):
        model = self.create_model()
        model.disj = Disjunction(expr=[
            [model.x.lor(model.y)], [model.y.lor(model.z)]
        ])
        with self.assertRaisesRegex(GDP_Error, "Found untransformed logical constraint.*"):
            TransformationFactory('gdp.bigm').create_using(model)
        with self.assertRaisesRegex(GDP_Error, "Found untransformed logical constraint.*"):
            TransformationFactory('gdp.hull').create_using(model)

    # TODO look to test_con.py for inspiration


if __name__ == "__main__":
    unittest.main()
