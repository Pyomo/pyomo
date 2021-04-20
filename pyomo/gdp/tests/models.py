from pyomo.core import (Block, ConcreteModel, Constraint, Objective, Param, Set,
                        Var, inequality, RangeSet, Any, Expression, maximize)
from pyomo.gdp import Disjunct, Disjunction


def oneVarDisj_2pts():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 10))
    m.disj1 = Disjunct()
    m.disj1.xTrue = Constraint(expr=m.x==1)
    m.disj2 = Disjunct()
    m.disj2.xFalse = Constraint(expr=m.x==0)
    m.disjunction = Disjunction(expr=[m.disj1, m.disj2])
    m.obj = Objective(expr=m.x)
    return m

def twoSegments_SawayaGrossmann():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 3))
    m.disj1 = Disjunct()
    m.disj1.c = Constraint(expr=inequality(0, m.x, 1))
    m.disj2 = Disjunct()
    m.disj2.c = Constraint(expr=inequality(2, m.x, 3))
    m.disjunction = Disjunction(expr=[m.disj1, m.disj2])

    # this is my objective because I want to make sure that when I am testing
    # cutting planes, my first solution to rBigM is not on the convex hull.
    m.obj = Objective(expr=m.x - m.disj2.indicator_var)

    return m

def makeTwoTermDisj():
    """Single two-term disjunction which has all of ==, <=, and >= constraints
    """
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
    """Single two-term disjunction which has all of ==, <=, and >= and 
    one nonlinear constraint.
    """
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
    """Single two-term disjunction with IndexedConstraints on both disjuncts.  
    Does not bound the variables, so cannot be transformed by hull at all and 
    requires specifying m values in bigm.
    """
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
    """Single two-term disjunction with IndexedConstraints on both disjuncts. 
    """
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

def localVar():
    """Two-term disjunction which declares a local variable y on one of the 
    disjuncts, which is used in the objective function as well.

    Used to test that we will treat y as global in the transformations, 
    despite where it is declared.
    """
    # y appears in a global constraint and a single disjunct.
    m = ConcreteModel()
    m.x = Var(bounds=(0,3))

    m.disj1 = Disjunct()
    m.disj1.cons = Constraint(expr=m.x >= 1)

    m.disj2 = Disjunct()
    m.disj2.y = Var(bounds=(1,3))
    m.disj2.cons = Constraint(expr=m.x + m.disj2.y == 3)

    m.disjunction = Disjunction(expr=[m.disj1, m.disj2])

    # This makes y global actually... But in disguise.
    m.objective = Objective(expr=m.x + m.disj2.y)
    return m


def makeThreeTermIndexedDisj():
    """Three-term indexed disjunction"""
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


def makeTwoTermDisj_boxes():
    m = ConcreteModel()
    m.x = Var(bounds=(0,5))
    m.y = Var(bounds=(0,5))
    def d_rule(disjunct, flag):
        m = disjunct.model()
        if flag:
            disjunct.c1 = Constraint(expr=inequality(1, m.x, 2))
            disjunct.c2 = Constraint(expr=inequality(3, m.y, 4))
        else:
            disjunct.c1 = Constraint(expr=inequality(3, m.x, 4))
            disjunct.c2 = Constraint(expr=inequality(1, m.y, 2))
    m.d = Disjunct([0,1], rule=d_rule)
    def disj_rule(m):
        return [m.d[0], m.d[1]]
    m.disjunction = Disjunction(rule=disj_rule)
    m.obj = Objective(expr=m.x + 2*m.y)
    return m


def makeThreeTermDisj_IndexedConstraints():
    """Three-term disjunction with indexed constraints on the disjuncts"""
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
    """Two-term indexed disjunction"""
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


def makeTwoTermIndexedDisjunction_BoundedVars():
    """Two-term indexed disjunction.
    Adds nothing to above--exists for historic reasons"""
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


def makeIndexedDisjunction_SkipIndex():
    """Two-term indexed disjunction where one of the two indices is skipped"""
    m = ConcreteModel()
    m.x = Var(bounds=(0, 10))
    @m.Disjunct([0,1])
    def disjuncts(d, i):
        m = d.model()
        d.cons = Constraint(expr=m.x == i)

    @m.Disjunction([0,1])
    def disjunctions(m, i):
        if i == 0:
            return Disjunction.Skip
        return [m.disjuncts[i], m.disjuncts[0]]

    return m

def makeTwoTermMultiIndexedDisjunction():
    """Two-term indexed disjunction with tuple indices"""
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


def makeTwoTermDisjOnBlock():
    """Two-term SimpleDisjunction on a block"""
    m = ConcreteModel()
    m.b = Block()
    m.a = Var(bounds=(0, 5))

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

def add_disj_not_on_block(m):
    def simpdisj_rule(disjunct):
        m = disjunct.model()
        disjunct.c = Constraint(expr=m.a >= 3)
    m.simpledisj = Disjunct(rule=simpdisj_rule)
    def simpledisj2_rule(disjunct):
        m = disjunct.model()
        disjunct.c = Constraint(expr=m.a <= 3.5)
    m.simpledisj2 = Disjunct(rule=simpledisj2_rule)
    m.disjunction2 = Disjunction(expr=[m.simpledisj, m.simpledisj2])
    return m

def makeDisjunctionsOnIndexedBlock():
    """Two disjunctions (one indexed an one not), each on a separate 
    BlockData of an IndexedBlock of length 2
    """
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
    """SimpleDisjunction where one of the Disjuncts contains three different 
    blocks: two simple and one indexed"""
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
    """Three-term SimpleDisjunction built from two IndexedDisjuncts and one
    SimpleDisjunct. The SimpleDisjunct and one of the DisjunctDatas each
    contain a nested SimpleDisjunction (the disjuncts of which are declared 
    on the same disjunct as the disjunction).

    (makeNestedDisjunctions_NestedDisjuncts is a much simpler model. All 
    this adds is that it has a nested disjunction on a DisjunctData as well
    as on a SimpleDisjunct. So mostly it exists for historical reasons.)
    """
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
    """Two-term SimpleDisjunction where one of the disjuncts contains a nested
    SimpleDisjunction, the disjuncts of which are declared on the model"""
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
    """Same as makeNestedDisjunctions_FlatDisjuncts except that the disjuncts
    of the nested disjunction are declared on the parent disjunct."""
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


def makeTwoSimpleDisjunctions():
    """Two SimpleDisjunctions on the same model."""
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
    m.disjunction2 = Disjunction(expr=[m.disjunct2[0], m.disjunct2[1]])
    return m


def makeDisjunctInMultipleDisjunctions():
    """This is not a transformable model! Two SimpleDisjunctions which have 
    a shared disjunct.
    """
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


def makeDuplicatedNestedDisjunction():
    """Not a transformable model (because of disjuncts shared between 
    disjunctions): A SimpleDisjunction where one of the disjuncts contains
    two SimpleDisjunctions with the same Disjuncts. This is a lazy
    way to test that we complain about untransformed disjunctions we encounter
    while transforming a disjunct.
    """
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
    """Two-term SimpleDisjunction where one of the disjuncts contains a 
    RangeSet"""
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1))
    m.d1 = Disjunct()
    m.d1.s = RangeSet(1)
    m.d1.c = Constraint(rule=lambda _: m.x == 1)
    m.d2 = Disjunct()
    m.disj = Disjunction(expr=[m.d1, m.d2])
    return m

##########################
# Grossmann lecture models
##########################

def grossmann_oneDisj():
    m = ConcreteModel()
    m.x = Var(bounds=(0,20))
    m.y = Var(bounds=(0, 20))
    m.disjunct1 = Disjunct()
    m.disjunct1.constraintx = Constraint(expr=inequality(0, m.x, 2))
    m.disjunct1.constrainty = Constraint(expr=inequality(7, m.y, 10))

    m.disjunct2 = Disjunct()
    m.disjunct2.constraintx = Constraint(expr=inequality(8, m.x, 10))
    m.disjunct2.constrainty = Constraint(expr=inequality(0, m.y, 3))

    m.disjunction = Disjunction(expr=[m.disjunct1, m.disjunct2])

    m.objective = Objective(expr=m.x + 2*m.y, sense=maximize)

    return m

def to_break_constraint_tolerances():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 130))
    m.y = Var(bounds=(0, 130))
    m.disjunct1 = Disjunct()
    m.disjunct1.constraintx = Constraint(expr=inequality(0, m.x, 2))
    m.disjunct1.constrainty = Constraint(expr=inequality(117, m.y, 127))

    m.disjunct2 = Disjunct()
    m.disjunct2.constraintx = Constraint(expr=inequality(118, m.x, 120))
    m.disjunct2.constrainty = Constraint(expr=inequality(0, m.y, 3))

    m.disjunction = Disjunction(expr=[m.disjunct1, m.disjunct2])

    m.objective = Objective(expr=m.x + 2*m.y, sense=maximize)

    return m

def grossmann_twoDisj():
    m = grossmann_oneDisj()

    m.disjunct3 = Disjunct()
    m.disjunct3.constraintx = Constraint(expr=inequality(1, m.x, 2.5))
    m.disjunct3.constrainty = Constraint(expr=inequality(6.5, m.y, 8))
    
    m.disjunct4 = Disjunct()
    m.disjunct4.constraintx = Constraint(expr=inequality(9, m.x, 11))
    m.disjunct4.constrainty = Constraint(expr=inequality(2, m.y, 3.5))

    m.disjunction2 = Disjunction(expr=[m.disjunct3, m.disjunct4])
    
    return m

def twoDisj_twoCircles_easy():
    m = ConcreteModel()
    m.x = Var(bounds=(0,8))
    m.y = Var(bounds=(0,10))

    m.upper_circle = Disjunct()
    m.upper_circle.cons = Constraint(expr=(m.x - 1)**2 + (m.y - 6)**2 <= 2)
    m.lower_circle = Disjunct()
    m.lower_circle.cons = Constraint(expr=(m.x - 4)**2 + (m.y - 2)**2 <= 2)

    m.disjunction = Disjunction(expr=[m.upper_circle, m.lower_circle])
    
    m.obj = Objective(expr=m.x + m.y, sense=maximize)
    return m

def fourCircles():
    m = twoDisj_twoCircles_easy()

    # and add two more overlapping circles, a la the Grossmann test case with
    # the rectangles. (but not change my nice integral optimal solution...)
    m.upper_circle2 = Disjunct()
    m.upper_circle2.cons = Constraint(expr=(m.x - 2)**2 + (m.y - 7)**2 <= 1)

    m.lower_circle2 = Disjunct()
    m.lower_circle2.cons = Constraint(expr=(m.x - 5)**2 + (m.y - 3)**2 <= 2)

    m.disjunction2 = Disjunction(expr=[m.upper_circle2, m.lower_circle2])

    return m

def makeDisjunctWithExpression():
    """Two-term SimpleDisjunction where one of the disjuncts contains an 
    Expression. This is used to make sure that we correctly handle types we 
    hit in disjunct.component_objects(active=True)"""
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1))
    m.d1 = Disjunct()
    m.d1.e = Expression(expr=m.x**2)
    m.d1.c = Constraint(rule=lambda _: m.x == 1)
    m.d2 = Disjunct()
    m.disj = Disjunction(expr=[m.d1, m.d2])
    return m

def makeDisjunctionOfDisjunctDatas():
    """Two SimpleDisjunctions, where each are disjunctions of DisjunctDatas.
    This adds nothing to makeTwoSimpleDisjunctions but exists for convenience
    because it has the same mathematical meaning as 
    makeAnyIndexedDisjunctionOfDisjunctDatas
    """
    m = ConcreteModel()
    m.x = Var(bounds=(-100, 100))

    m.obj = Objective(expr=m.x)

    m.idx = Set(initialize=[1,2])
    m.firstTerm = Disjunct(m.idx)
    m.firstTerm[1].cons = Constraint(expr=m.x == 0)
    m.firstTerm[2].cons = Constraint(expr=m.x == 2)
    m.secondTerm = Disjunct(m.idx)
    m.secondTerm[1].cons = Constraint(expr=m.x >= 2)
    m.secondTerm[2].cons = Constraint(expr=m.x >= 3)
    
    m.disjunction = Disjunction(expr=[m.firstTerm[1], m.secondTerm[1]])
    m.disjunction2 = Disjunction(expr=[m.firstTerm[2], m.secondTerm[2]])
    return m

def makeAnyIndexedDisjunctionOfDisjunctDatas():
    """An IndexedDisjunction indexed by Any, with two two-term DisjunctionDatas
    build from DisjunctDatas. Identical mathematically to 
    makeDisjunctionOfDisjunctDatas.

    Used to test that the right things happen for a case where soemone
    implements an algorithm which iteratively generates disjuncts and 
    retransforms"""
    m = ConcreteModel()
    m.x = Var(bounds=(-100, 100))

    m.obj = Objective(expr=m.x)

    m.idx = Set(initialize=[1,2])
    m.firstTerm = Disjunct(m.idx)
    m.firstTerm[1].cons = Constraint(expr=m.x == 0)
    m.firstTerm[2].cons = Constraint(expr=m.x == 2)
    m.secondTerm = Disjunct(m.idx)
    m.secondTerm[1].cons = Constraint(expr=m.x >= 2)
    m.secondTerm[2].cons = Constraint(expr=m.x >= 3)
    
    m.disjunction = Disjunction(Any)
    m.disjunction[1] = [m.firstTerm[1], m.secondTerm[1]]
    m.disjunction[2] = [m.firstTerm[2], m.secondTerm[2]]
    return m
