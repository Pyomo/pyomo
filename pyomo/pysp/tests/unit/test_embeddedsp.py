#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
import pyutilib.th as unittest

import pyomo.core as pc
from pyomo.pysp.embeddedsp import (EmbeddedSP,
                                   StageCostAnnotation,
                                   VariableStageAnnotation,
                                   StochasticDataAnnotation,
                                   Distribution,
                                   TableDistribution,
                                   UniformDistribution,
                                   NormalDistribution,
                                   LogNormalDistribution,
                                   GammaDistribution,
                                   BetaDistribution)
from pyomo.environ import (ConcreteModel,
                           Var,
                           ConstraintList,
                           Objective,
                           Expression,
                           Param)

class TestEmbeddedSP(unittest.TestCase):

    def test_collect_mutable_parameters(self):
        model = pc.ConcreteModel()
        model.p = pc.Param(mutable=True)
        model.q = pc.Param([1], mutable=True, initialize=1.0)
        model.r = pc.Param(initialize=1.1, mutable=False)
        model.x = pc.Var()
        for obj in [model.p, model.q[1]]:

            result = EmbeddedSP._collect_mutable_parameters(
                obj)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_mutable_parameters(
                obj + 1)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_mutable_parameters(
                2 * (obj + 1))
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_mutable_parameters(
                2 * obj)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_mutable_parameters(
                2 * obj + 1)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_mutable_parameters(
                2 * obj + 1 + model.x)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_mutable_parameters(
                obj * model.x)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_mutable_parameters(
                model.x / obj)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_mutable_parameters(
                model.x / (2 * obj))
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_mutable_parameters(
                obj * pc.log(2 * model.x))
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_mutable_parameters(
                obj * pc.sin(model.r) ** model.x)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_mutable_parameters(
                model.x**(obj * pc.sin(model.r)))
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

        result = EmbeddedSP._collect_mutable_parameters(
            1.0)
        self.assertEqual(len(result), 0)
        del result

        result = EmbeddedSP._collect_mutable_parameters(
            model.p + model.q[1] + model.r)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = EmbeddedSP._collect_mutable_parameters(
            model.p + 1 + model.r + model.q[1])
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)

        result = EmbeddedSP._collect_mutable_parameters(
            model.q[1] * 2 * (model.p + model.r) + model.r)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = EmbeddedSP._collect_mutable_parameters(
            2 * model.x * model.p * model.q[1] * model.r)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = EmbeddedSP._collect_mutable_parameters(
            2 * obj * model.q[1] * model.r + 1)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 1)
        del result

        result = EmbeddedSP._collect_mutable_parameters(
            2 * model.q[1] + 1 + model.x - model.p)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = EmbeddedSP._collect_mutable_parameters(
            model.r * model.x)
        self.assertEqual(len(result), 0)
        del result

        result = EmbeddedSP._collect_mutable_parameters(
            model.x / obj)
        self.assertTrue(id(obj) in result)
        self.assertEqual(len(result), 1)
        del result

        result = EmbeddedSP._collect_mutable_parameters(
            model.x / (2 * model.q[1] / model.p))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = EmbeddedSP._collect_mutable_parameters(
            (model.p / model.q[1]) * pc.log(2 * model.x))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = EmbeddedSP._collect_mutable_parameters(
            model.q[1] * pc.sin(model.p) ** (model.x + model.r))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = EmbeddedSP._collect_mutable_parameters(
            (model.p + model.x) ** (model.q[1] * pc.sin(model.r)))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

    def test_collect_variables(self):
        model = pc.ConcreteModel()
        model.p = pc.Var()
        model.p.fixed = True
        model.q = pc.Var([1])
        model.r = pc.Param(mutable=True)
        model.x = pc.Var()
        for obj in [model.p, model.q[1]]:

            result = EmbeddedSP._collect_variables(
                obj)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_variables(
                obj + 1)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_variables(
                2 * (obj + 1))
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_variables(
                2 * obj)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_variables(
                2 * obj + 1)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = EmbeddedSP._collect_variables(
                2 * obj + 1 + model.x)
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

            result = EmbeddedSP._collect_variables(
                obj * model.x)
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

            result = EmbeddedSP._collect_variables(
                model.x / obj)
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

            result = EmbeddedSP._collect_variables(
                model.x / (2 * obj))
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

            result = EmbeddedSP._collect_variables(
                obj * pc.log(2 * model.x))
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

            result = EmbeddedSP._collect_variables(
                obj * pc.sin(model.r) ** model.x)
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

            result = EmbeddedSP._collect_variables(
                model.x**(obj * pc.sin(model.r)))
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

        result = EmbeddedSP._collect_variables(
            1.0)
        self.assertEqual(len(result), 0)
        del result

        result = EmbeddedSP._collect_variables(
            model.p + model.q[1] + model.r)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = EmbeddedSP._collect_variables(
            model.p + 1 + model.r + model.q[1])
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)

        result = EmbeddedSP._collect_variables(
            model.q[1] * 2 * (model.p + model.r) + model.r)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = EmbeddedSP._collect_variables(
            2 * model.x * model.p * model.q[1] * model.r)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 3)
        del result

        result = EmbeddedSP._collect_variables(
            2 * obj * model.q[1] * model.r + 1)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 1)
        del result

        result = EmbeddedSP._collect_variables(
            2 * model.q[1] + 1 + model.x - model.p)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 3)
        del result

        result = EmbeddedSP._collect_variables(
            model.r * model.x)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 1)
        del result

        result = EmbeddedSP._collect_variables(
            model.x / obj)
        self.assertTrue(id(obj) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 2)
        del result

        result = EmbeddedSP._collect_variables(
            model.x / (2 * model.q[1] / model.p))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 3)
        del result

        result = EmbeddedSP._collect_variables(
            (model.p / model.q[1]) * pc.log(2 * model.x))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 3)
        del result

        result = EmbeddedSP._collect_variables(
            model.q[1] * pc.sin(model.p) ** (model.x + model.r))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 3)
        del result

        result = EmbeddedSP._collect_variables(
            (model.p + model.x) ** (model.q[1] * pc.sin(model.r)))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 3)
        del result

    def test_compute_time_stage(self):
        model = ConcreteModel()
        model.x = Var()
        model.y = Var([0,1])
        model.z = Var()
        model.p = Param(mutable=True)
        model.cost = Expression([0,1])
        model.cost[0] = model.x + model.y[0]
        model.cost[1] = model.p + model.y[1] + model.y[0]*model.p
        model.o = Objective(expr=model.cost[0] + model.cost[1])
        model.c = ConstraintList()
        model.c.add(model.x >= 1)               # 1
        model.c.add(model.y[0] >= 1)            # 2
        model.c.add(model.p * model.y[0] >= 1)  # 3
        model.c.add(model.y[0] >= model.p)      # 4
        model.c.add(model.p <= model.y[1])      # 5
        model.c.add(model.y[1] <= 1)            # 6
        model.c.add(model.x >= model.p)         # 7
        model.c.add(model.z == 1)               # 8

        model.varstage = VariableStageAnnotation()
        model.varstage.declare(model.x, 1)
        model.varstage.declare(model.y[0], 1, derived=True)
        model.varstage.declare(model.y[1], 2)
        model.varstage.declare(model.z, 2, derived=True)

        model.stagecost = StageCostAnnotation()
        model.stagecost.declare(model.cost[0], 1)
        model.stagecost.declare(model.cost[1], 2)

        model.stochdata = StochasticDataAnnotation()
        model.stochdata.declare(model.p,
                                distribution=UniformDistribution(0, 1))
        sp = EmbeddedSP(model)

        #
        # check variables
        #
        self.assertEqual(sp.compute_time_stage(model.x),
                         min(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.x,
                                               derived_last_stage=True),
                         min(sp.time_stages))

        self.assertEqual(sp.compute_time_stage(model.y[0]),
                         min(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.y[0],
                                               derived_last_stage=True),
                         max(sp.time_stages))

        self.assertEqual(sp.compute_time_stage(model.y[1]),
                         max(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.y[1],
                                               derived_last_stage=True),
                         max(sp.time_stages))

        self.assertEqual(sp.compute_time_stage(model.z),
                         max(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.z,
                                               derived_last_stage=True),
                         max(sp.time_stages))

        #
        # check constraints
        #
        self.assertEqual(sp.compute_time_stage(model.c[1]),
                         min(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.c[1],
                                               derived_last_stage=True),
                         min(sp.time_stages))

        self.assertEqual(sp.compute_time_stage(model.c[2]),
                         min(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.c[2],
                                               derived_last_stage=True),
                         max(sp.time_stages))

        self.assertEqual(sp.compute_time_stage(model.c[3]),
                         max(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.c[3],
                                               derived_last_stage=True),
                         max(sp.time_stages))

        self.assertEqual(sp.compute_time_stage(model.c[4]),
                         max(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.c[4],
                                               derived_last_stage=True),
                         max(sp.time_stages))

        self.assertEqual(sp.compute_time_stage(model.c[5]),
                         max(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.c[5],
                                               derived_last_stage=True),
                         max(sp.time_stages))

        self.assertEqual(sp.compute_time_stage(model.c[6]),
                         max(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.c[6],
                                               derived_last_stage=True),
                         max(sp.time_stages))

        self.assertEqual(sp.compute_time_stage(model.c[7]),
                         max(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.c[7],
                                               derived_last_stage=True),
                         max(sp.time_stages))

        self.assertEqual(sp.compute_time_stage(model.c[8]),
                         max(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.c[8],
                                               derived_last_stage=True),
                         max(sp.time_stages))

        #
        # check objectives and expressions
        #
        self.assertEqual(sp.compute_time_stage(model.cost[0]),
                         min(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.cost[0],
                                               derived_last_stage=True),
                         max(sp.time_stages))

        self.assertEqual(sp.compute_time_stage(model.cost[1]),
                         max(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.cost[1],
                                               derived_last_stage=True),
                         max(sp.time_stages))

        self.assertEqual(sp.compute_time_stage(model.o),
                         max(sp.time_stages))
        self.assertEqual(sp.compute_time_stage(model.o,
                                               derived_last_stage=True),
                         max(sp.time_stages))

    def test_Distribution(self):
        d = Distribution()
        with self.assertRaises(NotImplementedError):
            d.sample()

    def test_TableDistrubtion(self):
        with self.assertRaises(ValueError):
            TableDistribution([])
        with self.assertRaises(ValueError):
            TableDistribution([1],weights=[])
        with self.assertRaises(ValueError):
            TableDistribution([1],weights=[0.5])
        d = TableDistribution([1,2,3])
        v = d.sample()
        self.assertTrue(v in d.values)
        d = TableDistribution([1,2], weights=[0.5, 0.5])
        v = d.sample()
        self.assertTrue(v in d.values)
        d = TableDistribution([1,2], weights=[1, 0])
        v = d.sample()
        self.assertEqual(v, 1)

    def test_UniformDistrubtion(self):
        d = UniformDistribution(1,10)
        v = d.sample()
        self.assertTrue(1 <= v <= 10)

    def test_NormalDistrubtion(self):
        d = NormalDistribution(0,1)
        d.sample()

    def test_LogNormalDistrubtion(self):
        d = LogNormalDistribution(0,1)
        d.sample()

    def test_GammaDistrubtion(self):
        d = GammaDistribution(1,1)
        d.sample()

    def test_BetaDistrubtion(self):
        d = BetaDistribution(1,1)
        d.sample()


if __name__ == "__main__":
    unittest.main()
