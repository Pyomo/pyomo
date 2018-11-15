#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Tests for SequentialDecomposition
#

import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.network import *
from types import MethodType

try:
    import numpy, networkx
    import_available = True
except ImportError:
    import_available = False

gams_available = SolverFactory('gams').available(exception_flag=False)

@unittest.skipIf(not import_available, "numpy or networkx not available")
class TestSequentialDecomposition(unittest.TestCase):

    def is_converged(self, arc, rel=False, tol=1.0E-5):
        eblock = arc.expanded_block
        for name in arc.src.vars:
            if arc.src.vars[name].is_indexed():
                for i in arc.src.vars[name]:
                    sval = dval = None
                    if arc.src.is_extensive(name):
                        evar = eblock.component(name)
                        if evar is not None:
                            sf = eblock.component("splitfrac")
                            if sf is not None:
                                sval = value(arc.src.vars[name][i] * sf)
                            dval = value(evar[i])
                    if sval is None:
                        sval = value(arc.src.vars[name][i])
                    if dval is None:
                        dval = value(arc.dest.vars[name][i])
                    diff = abs(sval - dval)
                    if rel:
                        diff = diff / sval
                    if diff > tol:
                        return False
            else:
                sval = dval = None
                if arc.src.is_extensive(name):
                    evar = eblock.component(name)
                    if evar is not None:
                        sf = eblock.component("splitfrac")
                        if sf is not None:
                            sval = value(arc.src.vars[name] * sf)
                        dval = value(evar)
                if sval is None:
                    sval = value(arc.src.vars[name])
                if dval is None:
                    dval = value(arc.dest.vars[name])
                diff = abs(sval - dval)
                if rel:
                    diff = diff / sval
                if diff > tol:
                    return False

        return True

    def intensive_equal(self, port, tol=1.0E-5, **kwds):
        for name in kwds:
            if port.vars[name].is_indexed():
                for i in kwds[name]:
                    if abs(value(port.vars[name][i] - kwds[name][i])) > tol:
                        return False
                    if abs(value(port.vars[name][i] - kwds[name][i])) > tol:
                        return False
            else:
                if abs(value(port.vars[name] - kwds[name])) > tol:
                    return False
                if abs(value(port.vars[name] - kwds[name])) > tol:
                    return False

        return True

    def simple_recycle_model(self):
        m = ConcreteModel()
        m.comps = Set(initialize=["A", "B", "C"])

        # Feed
        m.feed = Block()

        m.feed.flow_out = Var(m.comps)
        m.feed.temperature_out = Var()
        m.feed.pressure_out = Var()

        m.feed.expr_var_idx_out = Var(m.comps)
        @m.feed.Expression(m.comps)
        def expr_idx_out(b, i):
            return -b.expr_var_idx_out[i]

        m.feed.expr_var_out = Var()
        m.feed.expr_out = -m.feed.expr_var_out

        @m.feed.Port()
        def outlet(b):
            return dict(flow=b.flow_out, temperature=b.temperature_out,
                pressure=b.pressure_out, expr_idx=b.expr_idx_out,
                expr=b.expr_out)

        def initialize_feed(self):
            pass

        m.feed.initialize = MethodType(initialize_feed, m.feed)

        # Mixer
        m.mixer = Block()

        m.mixer.flow_in_side_1 = Var(m.comps)
        m.mixer.temperature_in_side_1 = Var()
        m.mixer.pressure_in_side_1 = Var()

        m.mixer.expr_var_idx_in_side_1 = Var(m.comps)
        @m.mixer.Expression(m.comps)
        def expr_idx_in_side_1(b, i):
            return -b.expr_var_idx_in_side_1[i]

        m.mixer.expr_var_in_side_1 = Var()
        m.mixer.expr_in_side_1 = -m.mixer.expr_var_in_side_1

        m.mixer.flow_in_side_2 = Var(m.comps)
        m.mixer.temperature_in_side_2 = Var()
        m.mixer.pressure_in_side_2 = Var()

        m.mixer.expr_var_idx_in_side_2 = Var(m.comps)
        @m.mixer.Expression(m.comps)
        def expr_idx_in_side_2(b, i):
            return -b.expr_var_idx_in_side_2[i]

        m.mixer.expr_var_in_side_2 = Var()
        m.mixer.expr_in_side_2 = -m.mixer.expr_var_in_side_2

        m.mixer.flow_out = Var(m.comps)
        m.mixer.temperature_out = Var()
        m.mixer.pressure_out = Var()

        m.mixer.expr_var_idx_out = Var(m.comps)
        @m.mixer.Expression(m.comps)
        def expr_idx_out(b, i):
            return -b.expr_var_idx_out[i]

        m.mixer.expr_var_out = Var()
        m.mixer.expr_out = -m.mixer.expr_var_out

        @m.mixer.Port()
        def inlet_side_1(b):
            return dict(flow=b.flow_in_side_1,
                temperature=b.temperature_in_side_1,
                pressure=b.pressure_in_side_1,
                expr_idx=b.expr_idx_in_side_1,
                expr=b.expr_in_side_1)

        @m.mixer.Port()
        def inlet_side_2(b):
            return dict(flow=b.flow_in_side_2,
                temperature=b.temperature_in_side_2,
                pressure=b.pressure_in_side_2,
                expr_idx=b.expr_idx_in_side_2,
                expr=b.expr_in_side_2)

        @m.mixer.Port()
        def outlet(b):
            return dict(flow=b.flow_out,
                temperature=b.temperature_out,
                pressure=b.pressure_out,
                expr_idx=b.expr_idx_out,
                expr=b.expr_out)

        def initialize_mixer(self):
            for i in self.flow_out:
                self.flow_out[i].value = \
                    value(self.flow_in_side_1[i] + self.flow_in_side_2[i])
            for i in self.expr_var_idx_out:
                self.expr_var_idx_out[i].value = \
                    value(self.expr_var_idx_in_side_1[i] +
                        self.expr_var_idx_in_side_2[i])
            self.expr_var_out.value = \
                value(self.expr_var_in_side_1 + self.expr_var_in_side_2)
            assert self.temperature_in_side_1 == self.temperature_in_side_2
            self.temperature_out.value = value(self.temperature_in_side_1)
            assert self.pressure_in_side_1 == self.pressure_in_side_2
            self.pressure_out.value = value(self.pressure_in_side_1)

        m.mixer.initialize = MethodType(initialize_mixer, m.mixer)

        # Pass through
        m.unit = Block()

        m.unit.flow_in = Var(m.comps)
        m.unit.temperature_in = Var()
        m.unit.pressure_in = Var()

        m.unit.expr_var_idx_in = Var(m.comps)
        @m.unit.Expression(m.comps)
        def expr_idx_in(b, i):
            return -b.expr_var_idx_in[i]

        m.unit.expr_var_in = Var()
        m.unit.expr_in = -m.unit.expr_var_in

        m.unit.flow_out = Var(m.comps)
        m.unit.temperature_out = Var()
        m.unit.pressure_out = Var()

        m.unit.expr_var_idx_out = Var(m.comps)
        @m.unit.Expression(m.comps)
        def expr_idx_out(b, i):
            return -b.expr_var_idx_out[i]

        m.unit.expr_var_out = Var()
        m.unit.expr_out = -m.unit.expr_var_out

        @m.unit.Port()
        def inlet(b):
            return dict(flow=b.flow_in, temperature=b.temperature_in,
                pressure=b.pressure_in, expr_idx=b.expr_idx_in,
                expr=b.expr_in)

        @m.unit.Port()
        def outlet(b):
            return dict(flow=b.flow_out, temperature=b.temperature_out,
                pressure=b.pressure_out, expr_idx=b.expr_idx_out,
                expr=b.expr_out)

        def initialize_unit(self):
            for i in self.flow_out:
                self.flow_out[i].value = value(self.flow_in[i])
            for i in self.expr_var_idx_out:
                self.expr_var_idx_out[i].value = value(self.expr_var_idx_in[i])
            self.expr_var_out.value = value(self.expr_var_in)
            self.temperature_out.value = value(self.temperature_in)
            self.pressure_out.value = value(self.pressure_in)

        m.unit.initialize = MethodType(initialize_unit, m.unit)

        # Splitter
        m.splitter = Block()

        m.splitter.flow_in = Var(m.comps)
        m.splitter.temperature_in = Var()
        m.splitter.pressure_in = Var()

        m.splitter.expr_var_idx_in = Var(m.comps)
        @m.splitter.Expression(m.comps)
        def expr_idx_in(b, i):
            return -b.expr_var_idx_in[i]

        m.splitter.expr_var_in = Var()
        m.splitter.expr_in = -m.splitter.expr_var_in

        m.splitter.flow_out_side_1 = Var(m.comps)
        m.splitter.temperature_out_side_1 = Var()
        m.splitter.pressure_out_side_1 = Var()

        m.splitter.expr_var_idx_out_side_1 = Var(m.comps)
        @m.splitter.Expression(m.comps)
        def expr_idx_out_side_1(b, i):
            return -b.expr_var_idx_out_side_1[i]

        m.splitter.expr_var_out_side_1 = Var()
        m.splitter.expr_out_side_1 = -m.splitter.expr_var_out_side_1

        m.splitter.flow_out_side_2 = Var(m.comps)
        m.splitter.temperature_out_side_2 = Var()
        m.splitter.pressure_out_side_2 = Var()

        m.splitter.expr_var_idx_out_side_2 = Var(m.comps)
        @m.splitter.Expression(m.comps)
        def expr_idx_out_side_2(b, i):
            return -b.expr_var_idx_out_side_2[i]

        m.splitter.expr_var_out_side_2 = Var()
        m.splitter.expr_out_side_2 = -m.splitter.expr_var_out_side_2

        @m.splitter.Port()
        def inlet(b):
            return dict(flow=b.flow_in,
                temperature=b.temperature_in,
                pressure=b.pressure_in,
                expr_idx=b.expr_idx_in,
                expr=b.expr_in)

        @m.splitter.Port()
        def outlet_side_1(b):
            return dict(flow=b.flow_out_side_1,
                temperature=b.temperature_out_side_1,
                pressure=b.pressure_out_side_1,
                expr_idx=b.expr_idx_out_side_1,
                expr=b.expr_out_side_1)

        @m.splitter.Port()
        def outlet_side_2(b):
            return dict(flow=b.flow_out_side_2,
                temperature=b.temperature_out_side_2,
                pressure=b.pressure_out_side_2,
                expr_idx=b.expr_idx_out_side_2,
                expr=b.expr_out_side_2)

        def initialize_splitter(self):
            recycle = 0.1
            prod = 1 - recycle
            for i in self.flow_in:
                self.flow_out_side_1[i].value = prod * value(self.flow_in[i])
                self.flow_out_side_2[i].value = recycle * value(self.flow_in[i])
            for i in self.expr_var_idx_in:
                self.expr_var_idx_out_side_1[i].value = \
                    prod * value(self.expr_var_idx_in[i])
                self.expr_var_idx_out_side_2[i].value = \
                    recycle * value(self.expr_var_idx_in[i])
            self.expr_var_out_side_1.value = prod * value(self.expr_var_in)
            self.expr_var_out_side_2.value = recycle * value(self.expr_var_in)
            self.temperature_out_side_1.value = value(self.temperature_in)
            self.temperature_out_side_2.value = value(self.temperature_in)
            self.pressure_out_side_1.value = value(self.pressure_in)
            self.pressure_out_side_2.value = value(self.pressure_in)

        m.splitter.initialize = MethodType(initialize_splitter, m.splitter)

        # Prod
        m.prod = Block()

        m.prod.flow_in = Var(m.comps)
        m.prod.temperature_in = Var()
        m.prod.pressure_in = Var()

        m.prod.actual_var_idx_in = Var(m.comps)
        m.prod.actual_var_in = Var()

        @m.prod.Port()
        def inlet(b):
            return dict(flow=b.flow_in, temperature=b.temperature_in,
                pressure=b.pressure_in, expr_idx=b.actual_var_idx_in,
                expr=b.actual_var_in)

        def initialize_prod(self):
            pass

        m.prod.initialize = MethodType(initialize_prod, m.prod)

        # Arcs
        @m.Arc(directed=True)
        def stream_feed_to_mixer(m):
            return (m.feed.outlet, m.mixer.inlet_side_1)

        @m.Arc(directed=True)
        def stream_mixer_to_unit(m):
            return (m.mixer.outlet, m.unit.inlet)

        @m.Arc(directed=True)
        def stream_unit_to_splitter(m):
            return (m.unit.outlet, m.splitter.inlet)

        @m.Arc(directed=True)
        def stream_splitter_to_mixer(m):
            return (m.splitter.outlet_side_2, m.mixer.inlet_side_2)

        @m.Arc(directed=True)
        def stream_splitter_to_prod(m):
            return (m.splitter.outlet_side_1, m.prod.inlet)

        # Expand Arcs
        TransformationFactory("network.expand_arcs").apply_to(m)

        # Fix Feed
        m.feed.flow_out['A'].fix(100)
        m.feed.flow_out['B'].fix(200)
        m.feed.flow_out['C'].fix(300)
        m.feed.expr_var_idx_out['A'].fix(10)
        m.feed.expr_var_idx_out['B'].fix(20)
        m.feed.expr_var_idx_out['C'].fix(30)
        m.feed.expr_var_out.fix(40)
        m.feed.temperature_out.fix(450)
        m.feed.pressure_out.fix(128)

        return m

    def simple_recycle_run(self, tear_method, tol_type):
        rel = tol_type == "rel"
        m = self.simple_recycle_model()

        def function(unit):
            unit.initialize()

        seq = SequentialDecomposition(tear_method=tear_method,
            tol_type=tol_type)
        tset = [m.stream_splitter_to_mixer]
        seq.set_tear_set(tset)
        splitter_to_mixer_guess = {
            "flow": {"A": 0, "B": 0, "C": 0},
            "temperature": 450,
            "pressure": 128}
        seq.set_guesses_for(m.mixer.inlet_side_2, splitter_to_mixer_guess)
        # need to set guesses for expression members by initializing those vars
        m.mixer.expr_var_idx_in_side_2["A"] = 0
        m.mixer.expr_var_idx_in_side_2["B"] = 0
        m.mixer.expr_var_idx_in_side_2["C"] = 0
        m.mixer.expr_var_in_side_2 = 0
        seq.run(m, function)

        self.check_recycle_model(m, rel=rel)

    def extensive_recycle_model(self):
        def build_in_out(b):
            b.flow_in = Var(m.comps)
            b.mass_in = Var()
            b.temperature_in = Var()
            b.pressure_in = Var()

            b.expr_var_idx_in = Var(m.comps)
            @b.Expression(m.comps)
            def expr_idx_in(b, i):
                return -b.expr_var_idx_in[i]

            b.expr_var_in = Var()
            b.expr_in = -b.expr_var_in

            b.flow_out = Var(m.comps)
            b.mass_out = Var()
            b.temperature_out = Var()
            b.pressure_out = Var()

            b.expr_var_idx_out = Var(m.comps)
            @b.Expression(m.comps)
            def expr_idx_out(b, i):
                return -b.expr_var_idx_out[i]

            b.expr_var_out = Var()
            b.expr_out = -b.expr_var_out

            b.inlet = Port(rule=inlet)
            b.outlet = Port(rule=outlet)

            b.initialize = MethodType(initialize, b)

        def inlet(b):
            return dict(flow=(b.flow_in, Port.Extensive),
                mass=(b.mass_in, Port.Extensive),
                temperature=b.temperature_in, pressure=b.pressure_in,
                expr_idx=(b.expr_idx_in, Port.Extensive),
                expr=(b.expr_in, Port.Extensive))

        def outlet(b):
            return dict(flow=(b.flow_out, Port.Extensive),
                mass=(b.mass_out, Port.Extensive),
                temperature=b.temperature_out, pressure=b.pressure_out,
                expr_idx=(b.expr_idx_out, Port.Extensive),
                expr=(b.expr_out, Port.Extensive))

        def initialize(self):
            for i in self.flow_out:
                self.flow_out[i].value = value(self.flow_in[i])
            self.mass_out.value = value(self.mass_in)
            for i in self.expr_var_idx_out:
                self.expr_var_idx_out[i].value = value(self.expr_var_idx_in[i])
            self.expr_var_out.value = value(self.expr_var_in)
            self.temperature_out.value = value(self.temperature_in)
            self.pressure_out.value = value(self.pressure_in)

        def nop(self):
            pass

        m = ConcreteModel()
        m.comps = Set(initialize=["A", "B", "C"])

        # Feed
        m.feed = Block()

        m.feed.flow_out = Var(m.comps)
        m.feed.mass_out = Var()
        m.feed.temperature_out = Var()
        m.feed.pressure_out = Var()

        m.feed.expr_var_idx_out = Var(m.comps)
        @m.feed.Expression(m.comps)
        def expr_idx_out(b, i):
            return -b.expr_var_idx_out[i]

        m.feed.expr_var_out = Var()
        m.feed.expr_out = -m.feed.expr_var_out

        m.feed.outlet = Port(rule=outlet)

        m.feed.initialize = MethodType(nop, m.feed)

        # Mixer
        m.mixer = Block()
        build_in_out(m.mixer)

        # Pass through
        m.unit = Block()
        build_in_out(m.unit)

        # Splitter
        m.splitter = Block()
        build_in_out(m.splitter)

        # Prod
        m.prod = Block()

        m.prod.flow_in = Var(m.comps)
        m.prod.mass_in = Var()
        m.prod.temperature_in = Var()
        m.prod.pressure_in = Var()

        m.prod.actual_var_idx_in = Var(m.comps)
        m.prod.actual_var_in = Var()

        @m.prod.Port()
        def inlet(b):
            return dict(flow=(b.flow_in, Port.Extensive),
                mass=(b.mass_in, Port.Extensive),
                temperature=b.temperature_in, pressure=b.pressure_in,
                expr_idx=(b.actual_var_idx_in, Port.Extensive),
                expr=(b.actual_var_in, Port.Extensive))

        m.prod.initialize = MethodType(nop, m.prod)

        # Arcs
        @m.Arc(directed=True)
        def stream_feed_to_mixer(m):
            return (m.feed.outlet, m.mixer.inlet)

        @m.Arc(directed=True)
        def stream_mixer_to_unit(m):
            return (m.mixer.outlet, m.unit.inlet)

        @m.Arc(directed=True)
        def stream_unit_to_splitter(m):
            return (m.unit.outlet, m.splitter.inlet)

        @m.Arc(directed=True)
        def stream_splitter_to_mixer(m):
            return (m.splitter.outlet, m.mixer.inlet)

        @m.Arc(directed=True)
        def stream_splitter_to_prod(m):
            return (m.splitter.outlet, m.prod.inlet)

        # Split Fraction
        rec = 0.1
        prod = 1 - rec
        m.splitter.outlet.set_split_fraction(m.stream_splitter_to_mixer, rec)
        m.splitter.outlet.set_split_fraction(m.stream_splitter_to_prod, prod)

        # Expand Arcs
        TransformationFactory("network.expand_arcs").apply_to(m)

        # Fix Feed
        m.feed.flow_out['A'].fix(100)
        m.feed.flow_out['B'].fix(200)
        m.feed.flow_out['C'].fix(300)
        m.feed.mass_out.fix(400)
        m.feed.expr_var_idx_out['A'].fix(10)
        m.feed.expr_var_idx_out['B'].fix(20)
        m.feed.expr_var_idx_out['C'].fix(30)
        m.feed.expr_var_out.fix(40)
        m.feed.temperature_out.fix(450)
        m.feed.pressure_out.fix(128)

        return m

    def extensive_recycle_run(self, tear_method, tol_type):
        rel = tol_type == "rel"
        m = self.extensive_recycle_model()

        def function(unit):
            unit.initialize()

        seq = SequentialDecomposition(tear_method=tear_method,
            tol_type=tol_type)
        tset = [m.stream_splitter_to_mixer]
        seq.set_tear_set(tset)
        splitter_to_mixer_guess = {
            "flow": {"A": [(m.stream_splitter_to_mixer, 0)],
                     "B": [(m.stream_splitter_to_mixer, 0)],
                     "C": [(m.stream_splitter_to_mixer, 0)]},
            "mass": [(m.stream_splitter_to_mixer, 0)],
            "expr_idx": {"A": [(m.stream_splitter_to_mixer, 0)],
                         "B": [(m.stream_splitter_to_mixer, 0)],
                         "C": [(m.stream_splitter_to_mixer, 0)]},
            "expr": [(m.stream_splitter_to_mixer, 0)],
            "temperature": 450,
            "pressure": 128}
        seq.set_guesses_for(m.mixer.inlet, splitter_to_mixer_guess)
        seq.run(m, function)

        self.check_recycle_model(m, rel=rel)

        if rel:
            s = value(m.prod.inlet.mass)
            d = value(m.feed.outlet.mass)
            self.assertAlmostEqual(
                (s - d) / s, 0,
                places=5)
        else:
            self.assertAlmostEqual(
                value(m.prod.inlet.mass),
                value(m.feed.outlet.mass),
                places=5)

    def check_recycle_model(self, m, rel=False):
        for arc in m.component_data_objects(Arc):
            self.assertTrue(self.is_converged(arc, rel=rel))

        for port in m.component_data_objects(Port):
            self.assertTrue(self.intensive_equal(
                port,
                temperature=value(m.feed.outlet.temperature),
                pressure=value(m.feed.outlet.pressure)))

        if rel:
            # in == out
            for i in m.feed.outlet.flow:
                s = value(m.prod.inlet.flow[i])
                d = value(m.feed.outlet.flow[i])
                self.assertAlmostEqual(
                    (s - d) / s, 0,
                    places=5)
            for i in m.feed.outlet.expr_idx:
                s = value(m.prod.inlet.expr_idx[i])
                d = value(m.feed.outlet.expr_idx[i])
                self.assertAlmostEqual(
                    (s - d) / s, 0,
                    places=5)
            s = value(m.prod.inlet.expr)
            d = value(m.feed.outlet.expr)
            self.assertAlmostEqual(
                (s - d) / s, 0,
                places=5)

            # check the expressions work, should be negative in prod
            for i in m.feed.outlet.expr_idx:
                s = value(-m.prod.actual_var_idx_in[i])
                d = value(m.feed.expr_var_idx_out[i])
                self.assertAlmostEqual(
                    (s - d) / s, 0,
                    places=5)
            s = value(-m.prod.actual_var_in)
            d = value(m.feed.expr_var_out)
            self.assertAlmostEqual(
                (s - d) / s, 0,
                places=5)
        else:
            # in == out
            for i in m.feed.outlet.flow:
                self.assertAlmostEqual(
                    value(m.prod.inlet.flow[i]),
                    value(m.feed.outlet.flow[i]),
                    places=5)
            for i in m.feed.outlet.expr_idx:
                self.assertAlmostEqual(
                    value(m.prod.inlet.expr_idx[i]),
                    value(m.feed.outlet.expr_idx[i]),
                    places=5)
            self.assertAlmostEqual(
                value(m.prod.inlet.expr),
                value(m.feed.outlet.expr),
                places=5)

            # check the expressions work, should be negative in prod
            for i in m.feed.outlet.expr_idx:
                self.assertAlmostEqual(
                    value(-m.prod.actual_var_idx_in[i]),
                    value(m.feed.expr_var_idx_out[i]),
                    places=5)
            self.assertAlmostEqual(
                value(-m.prod.actual_var_in),
                value(m.feed.expr_var_out),
                places=5)

    def test_simple_recycle_direct_abs(self):
        self.simple_recycle_run(tear_method="Direct", tol_type="abs")

    def test_simple_recycle_wegstein_abs(self):
        self.simple_recycle_run(tear_method="Wegstein", tol_type="abs")

    def test_simple_recycle_direct_rel(self):
        self.simple_recycle_run(tear_method="Direct", tol_type="rel")

    def test_simple_recycle_wegstein_rel(self):
        self.simple_recycle_run(tear_method="Wegstein", tol_type="rel")

    def test_extensive_recycle_direct_abs(self):
        self.extensive_recycle_run(tear_method="Direct", tol_type="abs")

    def test_extensive_recycle_wegstein_abs(self):
        self.extensive_recycle_run(tear_method="Wegstein", tol_type="abs")

    def test_extensive_recycle_direct_rel(self):
        self.extensive_recycle_run(tear_method="Direct", tol_type="rel")

    def test_extensive_recycle_wegstein_rel(self):
        self.extensive_recycle_run(tear_method="Wegstein", tol_type="rel")

    @unittest.skipIf(not gams_available, "GAMS solver not available")
    def test_tear_selection(self):
        m = self.simple_recycle_model()
        seq = SequentialDecomposition()
        G = seq.create_graph(m)

        heu_result = seq.select_tear_heuristic(G)
        self.assertEqual(heu_result[1], 1)
        self.assertEqual(heu_result[2], 1)

        all_tsets = []
        for tset in heu_result[0]:
            all_tsets.append(seq.indexes_to_arcs(G, tset))
        for arc in (m.stream_mixer_to_unit, m.stream_unit_to_splitter,
                m.stream_splitter_to_mixer):
            self.assertIn([arc], all_tsets)

        tset_mip = seq.tear_set_arcs(G, "mip", solver="gams")
        self.assertIn(tset_mip, all_tsets)

        tset_heu = seq.tear_set_arcs(G, "heuristic")
        self.assertIn(tset_heu, all_tsets)

    @unittest.skipIf(not gams_available, "GAMS solver not available")
    def test_select_tear_in_run(self):
        m = self.simple_recycle_model()

        def function(unit):
            unit.initialize()

        seq = SequentialDecomposition()
        tset = [m.stream_splitter_to_mixer]
        seq.set_tear_set(tset)
        splitter_to_mixer_guess = {
            "flow": {"A": 0, "B": 0, "C": 0},
            "temperature": 450,
            "pressure": 128}
        seq.set_guesses_for(m.mixer.inlet_side_2, splitter_to_mixer_guess)
        # need to set guesses for expression members by initializing those vars
        m.mixer.expr_var_idx_in_side_2["A"] = 0
        m.mixer.expr_var_idx_in_side_2["B"] = 0
        m.mixer.expr_var_idx_in_side_2["C"] = 0
        m.mixer.expr_var_in_side_2 = 0
        seq.run(m, function)

        # we shouldn't need to know which streams are torn since everything
        # should already have values set so we don't need guesses, but we
        # just make sure it is able to select a tear set on its own
        seq = SequentialDecomposition(tear_solver="gams",
            select_tear_method="mip")
        seq.run(m, function)
        self.check_recycle_model(m)

        seq = SequentialDecomposition(tear_solver="gams",
            select_tear_method="heuristic")
        seq.run(m, function)
        self.check_recycle_model(m)

if __name__ == "__main__":
    unittest.main()
