from pyomo.core import (Block, ConcreteModel, Constraint, Objective, Param,
                        Set, Var, inequality, RangeSet)
from pyomo.gdp import Disjunct, Disjunction


def makeTwoTermDisj():
    m = ConcreteModel()
    m.a = Var(bounds=(2, 7))
    m.x = Var(bounds=(4, 9))

    def d_rule(disjunct, flag):
        m = disjunct.model()
        if flag:
            disjunct.c1 = Constraint(expr=m.a == 0)
            disjunct.c2 = Constraint(expr=m.x <= 7)
        else:
            disjunct.c = Constraint(expr=m.a >= 5)
    m.d = Disjunct([0, 1], rule=d_rule)
    m.disjunction = Disjunction(expr=[m.d[0], m.d[1]])
    return m


def makeTwoTermDisj_Nonlinear():
    m = ConcreteModel()
    m.w = Var(bounds=(2, 7))
    m.x = Var(bounds=(1, 8))
    m.y = Var(bounds=(-10, -3))

    def d_rule(disjunct, flag):
        m = disjunct.model()
        if flag:
            disjunct.c1 = Constraint(expr=m.x >= 2)
            disjunct.c2 = Constraint(expr=m.w == 3)
            disjunct.c3 = Constraint(expr=(1, m.x, 3))
        else:
            disjunct.c = Constraint(expr=m.x + m.y**2 <= 14)
    m.d = Disjunct([0, 1], rule=d_rule)
    m.disjunction = Disjunction(expr=[m.d[0], m.d[1]])
    return m


def makeTwoTermDisj_IndexedConstraints():
    m = ConcreteModel()
    m.s = Set(initialize=[1, 2])
    m.a = Var(m.s)
    m.b = Block()

    def disj1_rule(disjunct):
        m = disjunct.model()

        def c_rule(d, s):
            return m.a[s] == 0
        disjunct.c = Constraint(m.s, rule=c_rule)
    m.b.simpledisj1 = Disjunct(rule=disj1_rule)

    def disj2_rule(disjunct):
        m = disjunct.model()

        def c_rule(d, s):
            return m.a[s] <= 3
        disjunct.c = Constraint(m.s, rule=c_rule)
    m.b.simpledisj2 = Disjunct(rule=disj2_rule)
    m.b.disjunction = Disjunction(expr=[m.b.simpledisj1, m.b.simpledisj2])
    return m


def makeTwoTermDisj_IndexedConstraints_BoundedVars():
    # same concept as above, but bounded variables
    m = ConcreteModel()
    m.s = Set(initialize=[1, 2])
    m.lbs = Param(m.s, initialize={1: 2, 2: 4})
    m.ubs = Param(m.s, initialize={1: 7, 2: 6})

    def bounds_rule(m, s):
        return (m.lbs[s], m.ubs[s])
    m.a = Var(m.s, bounds=bounds_rule)

    def d_rule(disjunct, flag):
        m = disjunct.model()

        def true_rule(d, s):
            return m.a[s] == 0

        def false_rule(d, s):
            return m.a[s] >= 5
        if flag:
            disjunct.c = Constraint(m.s, rule=true_rule)
        else:
            disjunct.c = Constraint(m.s, rule=false_rule)
    m.disjunct = Disjunct([0, 1], rule=d_rule)
    m.disjunction = Disjunction(expr=[m.disjunct[0], m.disjunct[1]])
    return m


def makeThreeTermDisj_IndexedConstraints():
    m = ConcreteModel()
    m.I = [1, 2, 3]
    m.x = Var(m.I, bounds=(0, 10))

    def c_rule(b, i):
        m = b.model()
        return m.x[i] >= i

    def d_rule(d, j):
        m = d.model()
        d.c = Constraint(m.I[:j], rule=c_rule)
    m.d = Disjunct(m.I, rule=d_rule)
    m.disjunction = Disjunction(expr=[m.d[i] for i in m.I])
    return m


def makeTwoTermIndexedDisjunction():
    m = ConcreteModel()
    m.A = Set(initialize=[1, 2, 3])
    m.B = Set(initialize=['a', 'b'])
    m.x = Var(m.A, bounds=(-10, 10))

    def disjunct_rule(d, i, k):
        m = d.model()
        if k == 'a':
            d.cons_a = Constraint(expr=m.x[i] >= 5)
        if k == 'b':
            d.cons_b = Constraint(expr=m.x[i] <= 0)
    m.disjunct = Disjunct(m.A, m.B, rule=disjunct_rule)

    def disj_rule(m, i):
        return [m.disjunct[i, k] for k in m.B]
    m.disjunction = Disjunction(m.A, rule=disj_rule)
    return m


def makeTwoTermMultiIndexedDisjunction():
    m = ConcreteModel()
    m.s = Set(initialize=[1, 2])
    m.t = Set(initialize=['A', 'B'])
    m.a = Var(m.s, m.t, bounds=(2, 7))

    def d_rule(disjunct, flag, s, t):
        m = disjunct.model()
        if flag:
            disjunct.c = Constraint(expr=m.a[s, t] == 0)
        else:
            disjunct.c = Constraint(expr=m.a[s, t] >= 5)
    m.disjunct = Disjunct([0, 1], m.s, m.t, rule=d_rule)

    def disj_rule(m, s, t):
        return [m.disjunct[0, s, t], m.disjunct[1, s, t]]
    m.disjunction = Disjunction(m.s, m.t, rule=disj_rule)
    return m


def makeTwoTermIndexedDisjunction_BoundedVars():
    m = ConcreteModel()
    m.s = Set(initialize=[1, 2, 3])
    m.a = Var(m.s, bounds=(-100, 100))

    def disjunct_rule(d, s, flag):
        m = d.model()
        if flag:
            d.c = Constraint(expr=m.a[s] >= 6)
        else:
            d.c = Constraint(expr=m.a[s] <= 3)
    m.disjunct = Disjunct(m.s, [0, 1], rule=disjunct_rule)

    def disjunction_rule(m, s):
        return [m.disjunct[s, flag] for flag in [0, 1]]
    m.disjunction = Disjunction(m.s, rule=disjunction_rule)
    return m


def makeThreeTermIndexedDisj():
    m = ConcreteModel()
    m.s = Set(initialize=[1, 2])
    m.a = Var(m.s, bounds=(2, 7))

    def d_rule(disjunct, flag, s):
        m = disjunct.model()
        if flag == 0:
            disjunct.c = Constraint(expr=m.a[s] == 0)
        elif flag == 1:
            disjunct.c = Constraint(expr=m.a[s] >= 5)
        else:
            disjunct.c = Constraint(expr=inequality(2, m.a[s], 4))
    m.disjunct = Disjunct([0, 1, 2], m.s, rule=d_rule)

    def disj_rule(m, s):
        return [m.disjunct[0, s], m.disjunct[1, s], m.disjunct[2, s]]
    m.disjunction = Disjunction(m.s, rule=disj_rule)
    return m


def makeTwoTermDisjOnBlock():
    m = ConcreteModel()
    m.b = Block()
    m.a = Var(bounds=(0, 5))

    # On a whim, verify that the decorator notation works
    @m.b.Disjunct([0, 1])
    def disjunct(disjunct, flag):
        m = disjunct.model()
        if flag:
            disjunct.c = Constraint(expr=m.a <= 3)
        else:
            disjunct.c = Constraint(expr=m.a == 0)

    @m.b.Disjunction()
    def disjunction(m):
        return [m.disjunct[0], m.disjunct[1]]

    return m


def makeDisjunctionsOnIndexedBlock():
    m = ConcreteModel()
    m.s = Set(initialize=[1, 2])
    m.a = Var(m.s, bounds=(0, 70))

    @m.Disjunct(m.s, [0, 1])
    def disjunct1(disjunct, s, flag):
        m = disjunct.model()
        if not flag:
            disjunct.c = Constraint(expr=m.a[s] == 0)
        else:
            disjunct.c = Constraint(expr=m.a[s] >= 7)

    def disjunction1_rule(m, s):
        return [m.disjunct1[s, flag] for flag in [0, 1]]
    m.disjunction1 = Disjunction(m.s, rule=disjunction1_rule)

    m.b = Block([0, 1])
    m.b[0].x = Var(bounds=(-2, 2))

    def disjunct2_rule(disjunct, flag):
        if not flag:
            disjunct.c = Constraint(expr=m.b[0].x <= 0)
        else:
            disjunct.c = Constraint(expr=m.b[0].x >= 0)
    m.b[0].disjunct = Disjunct([0, 1], rule=disjunct2_rule)

    def disjunction(b, i):
        return [b.disjunct[0], b.disjunct[1]]
    m.b[0].disjunction = Disjunction([0], rule=disjunction)

    m.b[1].y = Var(bounds=(-3, 3))
    m.b[1].disjunct0 = Disjunct()
    m.b[1].disjunct0.c = Constraint(expr=m.b[1].y <= 0)
    m.b[1].disjunct1 = Disjunct()
    m.b[1].disjunct1.c = Constraint(expr=m.b[1].y >= 0)
    m.b[1].disjunction = Disjunction(
        expr=[m.b[1].disjunct0, m.b[1].disjunct1])
    return m


def makeTwoTermDisj_BlockOnDisj():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1000))
    m.y = Var(bounds=(0, 800))

    def disj_rule(d, flag):
        m = d.model()
        if flag:
            d.b = Block()
            d.b.c = Constraint(expr=m.x == 0)
            d.add_component('b.c', Constraint(expr=m.y >= 9))
            d.b.anotherblock = Block()
            d.b.anotherblock.c = Constraint(expr=m.y >= 11)
            d.bb = Block([1])
            d.bb[1].c = Constraint(expr=m.x == 0)
        else:
            d.c = Constraint(expr=m.x >= 80)
    m.evil = Disjunct([0, 1], rule=disj_rule)
    m.disjunction = Disjunction(expr=[m.evil[0], m.evil[1]])
    return m


def makeNestedDisjunctions():
    m = ConcreteModel()
    m.x = Var(bounds=(-9, 9))
    m.z = Var(bounds=(0, 10))
    m.a = Var(bounds=(0, 23))

    def disjunct_rule(disjunct, flag):
        m = disjunct.model()
        if flag:
            def innerdisj_rule(disjunct, flag):
                m = disjunct.model()
                if flag:
                    disjunct.c = Constraint(expr=m.z >= 5)
                else:
                    disjunct.c = Constraint(expr=m.z == 0)
            disjunct.innerdisjunct = Disjunct([0, 1], rule=innerdisj_rule)

            @disjunct.Disjunction([0])
            def innerdisjunction(b, i):
                return [b.innerdisjunct[0], b.innerdisjunct[1]]
            disjunct.c = Constraint(expr=m.a <= 2)
        else:
            disjunct.c = Constraint(expr=m.x == 2)
    m.disjunct = Disjunct([0, 1], rule=disjunct_rule)
    # I want a SimpleDisjunct with a disjunction in it too

    def simpledisj_rule(disjunct):
        m = disjunct.model()

        @disjunct.Disjunct()
        def innerdisjunct0(disjunct):
            disjunct.c = Constraint(expr=m.x <= 2)

        @disjunct.Disjunct()
        def innerdisjunct1(disjunct):
            disjunct.c = Constraint(expr=m.x >= 4)

        disjunct.innerdisjunction = Disjunction(
            expr=[disjunct.innerdisjunct0, disjunct.innerdisjunct1])
    m.simpledisjunct = Disjunct(rule=simpledisj_rule)
    m.disjunction = Disjunction(
        expr=[m.simpledisjunct, m.disjunct[0], m.disjunct[1]])
    return m


def makeNestedDisjunctions_FlatDisjuncts():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 2))
    m.obj = Objective(expr=m.x)
    m.d1 = Disjunct()
    m.d1.c = Constraint(expr=m.x >= 1)
    m.d2 = Disjunct()
    m.d2.c = Constraint(expr=m.x >= 1.1)
    m.d3 = Disjunct()
    m.d3.c = Constraint(expr=m.x >= 1.2)
    m.d4 = Disjunct()
    m.d4.c = Constraint(expr=m.x >= 1.3)
    m.disj = Disjunction(expr=[m.d1, m.d2])
    m.d1.disj = Disjunction(expr=[m.d3, m.d4])
    return m


def makeNestedDisjunctions_NestedDisjuncts():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 2))
    m.obj = Objective(expr=m.x)
    m.d1 = Disjunct()
    m.d1.c = Constraint(expr=m.x >= 1)
    m.d2 = Disjunct()
    m.d2.c = Constraint(expr=m.x >= 1.1)
    m.d1.d3 = Disjunct()
    m.d1.d3.c = Constraint(expr=m.x >= 1.2)
    m.d1.d4 = Disjunct()
    m.d1.d4.c = Constraint(expr=m.x >= 1.3)
    m.disj = Disjunction(expr=[m.d1, m.d2])
    m.d1.disj2 = Disjunction(expr=[m.d1.d3, m.d1.d4])
    return m


def makeDisjunctInMultipleDisjunctions():
    m = ConcreteModel()
    m.a = Var(bounds=(-10, 50))

    def d1_rule(disjunct, flag):
        m = disjunct.model()
        if flag:
            disjunct.c = Constraint(expr=m.a == 0)
        else:
            disjunct.c = Constraint(expr=m.a >= 5)
    m.disjunct1 = Disjunct([0, 1], rule=d1_rule)

    def d2_rule(disjunct, flag):
        if not flag:
            disjunct.c = Constraint(expr=m.a >= 30)
        else:
            disjunct.c = Constraint(expr=m.a == 100)
    m.disjunct2 = Disjunct([0, 1], rule=d2_rule)

    m.disjunction1 = Disjunction(expr=[m.disjunct1[0], m.disjunct1[1]])
    m.disjunction2 = Disjunction(expr=[m.disjunct2[0], m.disjunct1[1]])
    # Deactivate unused disjunct like we are supposed to
    m.disjunct2[1].deactivate()
    return m


def makeDisjunctInMultipleDisjunctions_no_deactivate():
    m = ConcreteModel()
    m.a = Var(bounds=(-10, 50))

    def d1_rule(disjunct, flag):
        m = disjunct.model()
        if flag:
            disjunct.c = Constraint(expr=m.a == 0)
        else:
            disjunct.c = Constraint(expr=m.a >= 5)
    m.disjunct1 = Disjunct([0, 1], rule=d1_rule)

    def d2_rule(disjunct, flag):
        if not flag:
            disjunct.c = Constraint(expr=m.a >= 30)
        else:
            disjunct.c = Constraint(expr=m.a == 100)
    m.disjunct2 = Disjunct([0, 1], rule=d2_rule)

    m.disjunction1 = Disjunction(expr=[m.disjunct1[0], m.disjunct1[1]])
    m.disjunction2 = Disjunction(expr=[m.disjunct2[0], m.disjunct1[1]])
    return m


def makeDuplicatedNestedDisjunction():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 8))

    def outerdisj_rule(d, flag):
        m = d.model()
        if flag:
            def innerdisj_rule(d, flag):
                m = d.model()
                if flag:
                    d.c = Constraint(expr=m.x >= 2)
                else:
                    d.c = Constraint(expr=m.x == 0)
            d.innerdisjunct = Disjunct([0, 1], rule=innerdisj_rule)
            d.innerdisjunction = Disjunction(expr=[d.innerdisjunct[0],
                                                   d.innerdisjunct[1]])
            d.duplicateddisjunction = Disjunction(expr=[d.innerdisjunct[0],
                                                        d.innerdisjunct[1]])
        else:
            d.c = Constraint(expr=m.x == 8)
    m.outerdisjunct = Disjunct([0, 1], rule=outerdisj_rule)
    m.disjunction = Disjunction(expr=[m.outerdisjunct[0],
                                      m.outerdisjunct[1]])
    return m


def makeDisjunctWithRangeSet():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1))
    m.d1 = Disjunct()
    m.d1.s = RangeSet(1)
    m.d1.c = Constraint(rule=lambda _: m.x == 1)
    m.d2 = Disjunct()
    m.disj = Disjunction(expr=[m.d1, m.d2])
    return m
