import pyutilib.th as unittest
from pyomo.environ import ConcreteModel, AbstractModel, BooleanVar, LogicalStatement


class TestLogicalStatementCreation(unittest.TestCase):
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
        model.p = LogicalStatement(rule=rule)

        self.assertIs(model.p.body, model.x)

    # TODO look to test_con.py for inspiration


if __name__ == "__main__":
    unittest.main()
