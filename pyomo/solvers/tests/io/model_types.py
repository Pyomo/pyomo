#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from os.path import join, dirname, abspath
import json

from pyomo.core import *
from pyomo.repn.beta.matrix import compile_block_linear_constraints

thisDir = dirname(abspath( __file__ ))

class _ModelClassBase(object):
    """
    The base class for all model classifications
    """
    def __init__(self):
        self.linear = False
        self.integer = False
        self.quadratic_objective = False
        self.quadratic_constraint = False
        self.sos1 = False
        self.sos2 = False
        self.model = None
        self.results_file = None
        self.disable_suffix_tests = False
        self.diff_tol = 1e-4

    def generateModel():
        raise NotImplementedError

    def saveCurrentSolution(self,filename,**kwds):
        assert self.model is not None
        model = self.model
        suffixes = dict((suffix, getattr(model,suffix))
                        for suffix in kwds.pop('suffixes',[]))
        for suf in suffixes.values():
            assert isinstance(suf,Suffix)
            assert suf.import_enabled()
        with open(filename,'w') as f:
            soln = {}
            for block in model.block_data_objects():
                soln[block.cname(True)] = {}
                for suffix_name, suffix in suffixes.items():
                    if suffix.get(block) is not None:
                        soln[block.cname(True)][suffix_name] = suffix.get(block)
            for var in model.component_data_objects(Var):
                soln[var.cname(True)] = {}
                soln[var.cname(True)]['value'] = var.value
                soln[var.cname(True)]['stale'] = var.stale
                for suffix_name, suffix in suffixes.items():
                    if suffix.get(var) is not None:
                        soln[var.cname(True)][suffix_name] = suffix.get(var)
            for con in model.component_data_objects(Constraint):
                soln[con.cname(True)] = {}
                con_value = con(exception=False)
                soln[con.cname(True)]['value'] = con_value
                for suffix_name, suffix in suffixes.items():
                    if suffix.get(con) is not None:
                        soln[con.cname(True)][suffix_name] = suffix.get(con)
            for obj in model.component_data_objects(Objective):
                soln[obj.cname(True)] = {}
                obj_value = obj(exception=False)
                soln[obj.cname(True)]['value'] = obj_value
                for suffix_name, suffix in suffixes.items():
                    if suffix.get(obj) is not None:
                        soln[obj.cname(True)][suffix_name] = suffix.get(obj)
            json.dump(soln, f, indent=2, sort_keys=True)

    def validateCurrentSolution(self,**kwds):
        assert self.model is not None
        assert self.results_file is not None
        model = self.model
        suffixes = dict((suffix, getattr(model,suffix))
                        for suffix in kwds.pop('suffixes',[]))
        for suf in suffixes.values():
            assert isinstance(suf,Suffix)
            assert suf.import_enabled()
        solution = None
        error_str = ("Difference in solution for {0}.{1}:\n\tBaseline "
                     "- {2}\n\tCurrent - {3}")
        with open(self.results_file,'r') as f:
            try:
                solution = json.load(f)
            except:
                return (False,"Problem reading file "+self.results_file)
        for var in model.component_data_objects(Var):
            var_value_sol = solution[var.cname(True)]['value']
            var_value = var.value
            if not ((var_value is None) and (var_value_sol is None)):
                if ((var_value is None) ^ (var_value_sol is None)) or \
                   (abs(var_value_sol - var_value) > self.diff_tol):
                    return (False,
                            error_str.format(var.cname(True),
                                             'value',
                                             var_value_sol,
                                             var_value))
            if not (solution[var.cname(True)]['stale'] is var.stale):
                return (False,
                        error_str.format(var.cname(True),
                                         'stale',
                                         solution[var.cname(True)]['stale'],
                                         var.stale))
            for suffix_name, suffix in suffixes.items():
                if suffix_name in solution[var.cname(True)]:
                    if suffix.get(var) is None:
                        if not(solution[var.cname(True)][suffix_name] in \
                               solution["suffix defaults"][suffix_name]):
                            return (False,
                                    error_str.format(
                                        var.cname(True),
                                        suffix,
                                        solution[var.cname(True)][suffix_name],
                                        "none defined"))
                    elif not abs(solution[var.cname(True)][suffix_name] - \
                                 suffix.get(var)) < self.diff_tol:
                        return (False,
                                error_str.format(
                                    var.cname(True),
                                    suffix,
                                    solution[var.cname(True)][suffix_name],
                                    suffix.get(var)))
        for con in model.component_data_objects(Constraint):
            con_value_sol = solution[con.cname(True)]['value']
            con_value = con(exception=False)
            if not ((con_value is None) and (con_value_sol is None)):
                if ((con_value is None) ^ (con_value_sol is None)) or \
                   (abs(con_value_sol - con_value) > self.diff_tol):
                    return (False,
                            error_str.format(con.cname(True),
                                             'value',
                                             con_value_sol,
                                             con_value))
            for suffix_name, suffix in suffixes.items():
                if suffix_name in solution[con.cname(True)]:
                    if suffix.get(con) is None:
                        if not (solution[con.cname(True)][suffix_name] in \
                                solution["suffix defaults"][suffix_name]):
                            return (False,
                                    error_str.format(
                                        con.cname(True),
                                        suffix,
                                        solution[con.cname(True)][suffix_name],
                                        "none defined"))
                    elif not abs(solution[con.cname(True)][suffix_name] - \
                                 suffix.get(con)) < self.diff_tol:
                        return (False,
                                error_str.format(
                                    con.cname(True),
                                    suffix,
                                    solution[con.cname(True)][suffix_name],
                                    suffix.get(con)))
        for obj in model.component_data_objects(Objective):
            obj_value_sol = solution[obj.cname(True)]['value']
            obj_value = obj(exception=False)
            if not ((obj_value is None) and (obj_value_sol is None)):
                if ((obj_value is None) ^ (obj_value_sol is None)) or \
                   (abs(obj_value_sol - obj_value) > self.diff_tol):
                    return (False,
                            error_str.format(obj.cname(True),
                                             'value',
                                             obj_value_sol,
                                             obj_value))
            for suffix_name, suffix in suffixes.items():
                if suffix_name in solution[obj.cname(True)]:
                    if suffix.get(obj) is None:
                        if not(solution[obj.cname(True)][suffix_name] in \
                               solution["suffix defaults"][suffix_name]):
                            return (False,
                                    error_str.format(
                                        obj.cname(True),
                                        suffix,
                                        solution[obj.cname(True)][suffix_name],
                                        "none defined"))
                    elif not abs(solution[obj.cname(True)][suffix_name] - \
                                 suffix.get(obj)) < self.diff_tol:
                        return (False,
                                error_str.format(
                                    obj.cname(True),
                                    suffix,
                                    solution[obj.cname(True)][suffix_name],
                                    suffix.get(obj)))
        for block in model.block_data_objects():
            for suffix_name, suffix in suffixes.items():
                if (solution[block.cname(True)] is not None) and \
                   (suffix_name in solution[block.cname(True)]):
                    if suffix.get(block) is None:
                        if not(solution[block.cname(True)][suffix_name] in \
                               solution["suffix defaults"][suffix_name]):
                            return (False,
                                    error_str.format(
                                        block.cname(True),
                                        suffix,
                                        solution[block.cname(True)][suffix_name],
                                        "none defined"))
                    elif not abs(solution[block.cname(True)][suffix_name] - \
                                 suffix.get(block)) < sefl.diff_tol:
                        return (False,
                                error_str.format(
                                    block.cname(True),
                                    suffix,
                                    solution[block.cname(True)][suffix_name],
                                    suffix.get(block)))
        return (True,"")

    def descrStr(self):
        raise NotImplementedError

    def validateCapabilities(self,opt):
        if (self.linear is True) and \
           (not opt.has_capability('linear') is True):
            return False
        if (self.integer is True) and \
           (not opt.has_capability('integer') is True):
            return False
        if (self.quadratic_objective is True) and \
           (not opt.has_capability('quadratic_objective') is True):
            return False
        if (self.quadratic_constraint is True) and \
           (not opt.has_capability('quadratic_constraint') is True):
            return False
        if (self.sos1 is True) and \
           (not opt.has_capability('sos1') is True):
            return False
        if (self.sos2 is True) and \
           (not opt.has_capability('sos2') is True):
            return False
        return True

    def disableSuffixTests(self):
        return self.disable_suffix_tests

    def postSolveTestValidation(self, tester, results):
        pass

# Define globally for pickling purposes
def inactive_index_LP_obj_rule(model,i):
    if i == 1:
        return model.x-model.y
    else:
        return -model.x+model.y+model.z

def inactive_index_LP_c2_rule(model,i):
    if i == 1:
        return model.y >= -2
    else:
        return model.x <= 2

class simple_LP(_ModelClassBase):
    """
    A continuous linear model
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.results_file = join(thisDir,"simple_LP.results")

    def descrStr(self):
        return "simple_LP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.a1 = Param(initialize=1.0, mutable=True)
        model.a2 = Param([1], initialize=1.0, mutable=True)
        model.a3 = Param(initialize=1.0, mutable=False)
        model.a4 = Param([1], initialize=1.0, mutable=False)
        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=NonNegativeReals)
        model.z1 = Var()
        model.z2 = Var()
        model.dummy_expr1 = Expression(initialize=model.a1*model.a2[1])
        model.dummy_expr2 = Expression(initialize=model.y/model.a3*model.a4[1])

        model.inactive_obj = Objective(expr=model.x + 3.0*model.y + 1.0 + model.z1 - model.z2)
        model.inactive_obj.deactivate()
        model.obj = Objective(expr=model.inactive_obj)
        model.c1 = Constraint(expr=model.dummy_expr1 <= model.dummy_expr2)
        model.c2 = Constraint(expr=2.0 <= model.x/model.a3 - model.y <= 10)
        model.c3 = Constraint(expr=0 <= model.z1 + 1 <= 10)
        model.c4 = Constraint(expr=-10 <= model.z2 + 1 <= 0)

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x = None
        model.y = 1.0

class trivial_constraints_LP(_ModelClassBase):
    """
    A continuous linear model with trivial constraints
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.results_file = join(thisDir,"trivial_constraints_LP.results")

    def descrStr(self):
        return "trivial_constraints_LP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.x = Var(domain=RealInterval(bounds=(float('-inf'), None)))
        model.y = Var(bounds=(None, float('inf')))
        model.obj = Objective(expr=model.x - model.y)
        model.c = ConstraintList()
        model.c.add(model.x >= -2)
        model.c.add(model.y <= 3)
        cdata = model.c.add((0, 1, 3))
        assert cdata.lower == 0
        assert cdata.upper == 3
        assert cdata.body() == 1
        assert not cdata.equality
        cdata = model.c.add((0, 2, 3))
        assert cdata.lower == 0
        assert cdata.upper == 3
        assert cdata.body() == 2
        assert not cdata.equality
        cdata = model.c.add((0, 1, None))
        assert cdata.lower == 0
        assert cdata.upper is None
        assert cdata.body() == 1
        assert not cdata.equality
        cdata = model.c.add((None, 0, 1))
        assert cdata.lower is None
        assert cdata.upper == 1
        assert cdata.body() == 0
        assert not cdata.equality
        cdata = model.c.add((1,1))
        assert cdata.lower == 1
        assert cdata.upper == 1
        assert cdata.body() == 1
        assert cdata.equality

    def warmstartModel(self):
        assert self.model is not None
        pass

    def postSolveTestValidation(self, tester, results):
        _ModelClassBase.postSolveTestValidation(self, tester, results)
        symbol_map = results._smap
        tester.assertNotEqual(symbol_map, None)
        for i in self.model.c:
            tester.assertTrue(id(self.model.c[i]) in symbol_map.byObject)

class piecewise_LP(_ModelClassBase):
    """
    A continuous linear model
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.results_file = join(thisDir,"piecewise_LP.results")

    def descrStr(self):
        return "piecewise_LP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.x = Var()
        model.y = Var()

        model.obj = Objective(expr=model.y)
        model.p = Piecewise(model.y, model.x,
                            pw_pts=[-1,0,1],
                            f_rule=[1,0.5,1],
                            pw_repn='SOS2',
                            pw_constr_type='LB',
                            unbounded_domain_var=True)

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x = None
        model.y = 1.0

class piecewise_LP_nosuffixes(piecewise_LP):
    def __init__(self):
        piecewise_LP.__init__(self)
        self.disable_suffix_tests = True

class constant_objective_LP1(_ModelClassBase):
    """
    A continuous linear model with a constant objective
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.results_file = join(thisDir,"constant_objective_LP1.results")

    def descrStr(self):
        return "constant_objective_LP1"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.x = Var(within=NonNegativeReals)
        model.obj = Objective(expr=0.0)
        model.con = Constraint(expr=model.x == 1.0)

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x = None

class constant_objective_LP2(_ModelClassBase):
    """
    A continuous linear model with a constant objective that
    starts as a linear expression
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.results_file = join(thisDir,"constant_objective_LP2.results")

    def descrStr(self):
        return "constant_objective_LP2"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.x = Var(within=NonNegativeReals)
        model.obj = Objective(expr=model.x-model.x)
        model.con = Constraint(expr=model.x == 1.0)

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x = 1.0

# NOTE: We could test this problem on solvers that only handle
#       linear objectives IF we could get some proper preprocessing
#       in place for the canonical_repn
class constant_objective_QP(_ModelClassBase):
    """
    A continuous linear model with a constant objective that starts
    as quadratic expression
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        # Delete to enable tests on linear objective only solvers
        self.quadratic_objective = True
        self.results_file = join(thisDir,"constant_objective_QP.results")

    def descrStr(self):
        return "constant_objective_QP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.x = Var(within=NonNegativeReals)
        model.obj = Objective(expr=model.x**2-model.x**2)
        model.con = Constraint(expr=model.x == 1.0)

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x = 1.0

class block_LP(_ModelClassBase):
    """
    A continuous linear model with nested blocks
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.results_file = join(thisDir,"block_LP.results")

    def descrStr(self):
        return "block_LP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.b = Block()
        model.B = Block([1,2,3])
        model.a = Param(initialize=1.0, mutable=True)
        model.b.x = Var(within=NonNegativeReals)
        model.B[1].x = Var(within=NonNegativeReals)

        model.obj = Objective(expr=model.b.x + 3.0*model.B[1].x)
        model.obj.deactivate()
        model.B[2].c = Constraint(expr=-model.B[1].x <= -model.a)
        model.B[2].obj = Objective(expr=model.b.x + 3.0*model.B[1].x + 2)
        model.B[3].c = Constraint(expr=2.0 <= model.b.x/model.a - model.B[1].x <= 10)

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.b.x = 1.0
        model.B[1].x = 1.0

class inactive_index_LP(_ModelClassBase):
    """
    A continuous linear model where component subindices have been deactivated
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.results_file = join(thisDir,"inactive_index_LP.results")

    def descrStr(self):
        return "inactive_index_LP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.s = Set(initialize=[1,2])
        model.x = Var()
        model.y = Var()
        model.z = Var(bounds=(0,None))

        model.obj = Objective(model.s,
                              rule=inactive_index_LP_obj_rule)
        model.OBJ = Objective(expr=model.x+model.y)
        model.obj[1].deactivate()
        model.OBJ.deactivate()
        model.c1 = ConstraintList()
        model.c1.add(model.x<=1)   # index=1
        model.c1.add(model.x>=-1)  # index=2
        model.c1.add(model.y<=1)   # index=3
        model.c1.add(model.y>=-1)  # index=4
        model.c1[1].deactivate()
        model.c1[4].deactivate()
        model.c2 = Constraint(model.s,
                              rule=inactive_index_LP_c2_rule)

        model.b = Block()
        model.b.c = Constraint(expr=model.z >= 2)
        model.B = Block(model.s)
        model.B[1].c = Constraint(expr=model.z >= 3)
        model.B[2].c = Constraint(expr=model.z >= 1)

        model.b.deactivate()
        model.B.deactivate()
        model.B[2].activate()

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x = None
        model.y = None
        model.z = 2.0

class unused_vars_LP(_ModelClassBase):
    """
    A continuous linear model where some vars aren't used
    and some used vars start out with the stale flag as True
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.disable_suffix_tests = True
        self.results_file = join(thisDir,"unused_vars_LP.results")

    def descrStr(self):
        return "unused_vars_LP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.s = Set(initialize=[1,2])

        model.x_unused = Var()
        model.x_unused.stale = False

        model.x_unused_initialy_stale = Var()
        model.x_unused_initialy_stale.stale = True

        model.X_unused = Var(model.s)
        model.X_unused_initialy_stale = Var(model.s)
        for i in model.s:
            model.X_unused[i].stale = False
            model.X_unused_initialy_stale[i].stale = True

        model.x = Var()
        model.x.stale = False

        model.x_initialy_stale = Var()
        model.x_initialy_stale.stale = True

        model.X = Var(model.s)
        model.X_initialy_stale = Var(model.s)
        for i in model.s:
            model.X[i].stale = False
            model.X_initialy_stale[i].stale = True

        model.obj = Objective(expr= model.x + \
                                    model.x_initialy_stale + \
                                    summation(model.X) + \
                                    summation(model.X_initialy_stale))

        model.c = ConstraintList()
        model.c.add( model.x          >= 1 )
        model.c.add( model.x_initialy_stale    >= 1 )
        model.c.add( model.X[1]       >= 0 )
        model.c.add( model.X[2]       >= 1 )
        model.c.add( model.X_initialy_stale[1] >= 0 )
        model.c.add( model.X_initialy_stale[2] >= 1 )

        # Test that stale flags do not get updated
        # on inactive blocks (where "inactive blocks" mean blocks
        # that do NOT follow a path of all active parent blocks
        # up to the top-level model)
        flat_model = model.clone()
        model.b = Block()
        model.B = Block(model.s)
        model.b.b = flat_model.clone()
        model.B[1].b = flat_model.clone()
        model.B[2].b = flat_model.clone()

        model.b.deactivate()
        model.B.deactivate()
        model.b.b.activate()
        model.B[1].b.activate()
        model.B[2].b.deactivate()
        assert model.b.active is False
        assert model.B[1].active is False
        assert model.B[1].active is False
        assert model.b.b.active is True
        assert model.B[1].b.active is True
        assert model.B[2].b.active is False

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x_unused = -1.0
        model.x_unused_initialy_stale = -1.0
        for i in model.s:
            model.X_unused[i] = -1.0
            model.X_unused_initialy_stale[i] = -1.0

        model.x = -1.0
        model.x_initialy_stale = -1.0
        for i in model.s:
            model.X[i] = -1.0
            model.X_initialy_stale[i] = -1.0

class unused_vars_MILP(_ModelClassBase):
    """
    A continuous linear model where some vars aren't used
    and some used vars start out with the stale flag as True
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.disable_suffix_tests = True
        self.results_file = join(thisDir,"unused_vars_MILP.results")

    def descrStr(self):
        return "unused_vars_MILP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.s = Set(initialize=[1,2])

        model.x_unused = Var(within=Integers)
        model.x_unused.stale = False

        model.x_unused_initialy_stale = Var(within=Integers)
        model.x_unused_initialy_stale.stale = True

        model.X_unused = Var(model.s, within=Integers)
        model.X_unused_initialy_stale = Var(model.s, within=Integers)
        for i in model.s:
            model.X_unused[i].stale = False
            model.X_unused_initialy_stale[i].stale = True

        model.x = Var(within=IntegerInterval(bounds=(None,None)))
        model.x.stale = False

        model.x_initialy_stale = Var(within=Integers)
        model.x_initialy_stale.stale = True

        model.X = Var(model.s, within=Integers)
        model.X_initialy_stale = Var(model.s, within=Integers)
        for i in model.s:
            model.X[i].stale = False
            model.X_initialy_stale[i].stale = True

        model.obj = Objective(expr= model.x + \
                                    model.x_initialy_stale + \
                                    summation(model.X) + \
                                    summation(model.X_initialy_stale))

        model.c = ConstraintList()
        model.c.add( model.x          >= 1 )
        model.c.add( model.x_initialy_stale    >= 1 )
        model.c.add( model.X[1]       >= 0 )
        model.c.add( model.X[2]       >= 1 )
        model.c.add( model.X_initialy_stale[1] >= 0 )
        model.c.add( model.X_initialy_stale[2] >= 1 )

        # Test that stale flags do not get updated
        # on inactive blocks (where "inactive blocks" mean blocks
        # that do NOT follow a path of all active parent blocks
        # up to the top-level model)
        flat_model = model.clone()
        model.b = Block()
        model.B = Block(model.s)
        model.b.b = flat_model.clone()
        model.B[1].b = flat_model.clone()
        model.B[2].b = flat_model.clone()

        model.b.deactivate()
        model.B.deactivate()
        model.b.b.activate()
        model.B[1].b.activate()
        model.B[2].b.deactivate()
        assert model.b.active is False
        assert model.B[1].active is False
        assert model.B[1].active is False
        assert model.b.b.active is True
        assert model.B[1].b.active is True
        assert model.B[2].b.active is False

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x_unused = -1
        model.x_unused_initialy_stale = -1
        for i in model.s:
            model.X_unused[i] = -1
            model.X_unused_initialy_stale[i] = -1

        model.x = -1
        model.x_initialy_stale = -1
        for i in model.s:
            model.X[i] = -1
            model.X_initialy_stale[i] = -1

class simple_MILP(_ModelClassBase):
    """
    A mixed-integer linear model
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.integer = True
        self.results_file = join(thisDir,"simple_MILP.results")

    def descrStr(self):
        return "simple_MILP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.a = Param(initialize=1.0)
        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=Binary)

        model.obj = Objective(expr=model.x + 3.0*model.y)
        model.c1 = Constraint(expr=model.a <= model.y)
        model.c2 = Constraint(expr=2.0 <= model.x/model.a - model.y <= 10)

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x = 0.1
        model.y = 0

class discrete_var_bounds_MILP(_ModelClassBase):
    """
    A discrete model where discrete variables have custom bounds
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.integer = True
        self.disable_suffix_tests = True
        self.results_file = join(thisDir,"discrete_var_bounds_MILP.results")

    def descrStr(self):
        return "discrete_var_bounds_MILP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.w2 = Var(within=Binary)
        model.x2 = Var(within=Binary)
        model.yb = Var(within=Binary, bounds=(1,1))
        model.zb = Var(within=Binary, bounds=(0,0))
        model.yi = Var(within=Integers, bounds=(-1,None))
        model.zi = Var(within=Integers, bounds=(None,1))

        model.obj = Objective(expr=\
                                  model.w2 - model.x2 +\
                                  model.yb - model.zb +\
                                  model.yi - model.zi)

        model.c3 = Constraint(expr=model.w2 >= 0)
        model.c4 = Constraint(expr=model.x2 <= 1)

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.w2 = None
        model.x2 = 1
        model.yb = 0
        model.zb = 1
        model.yi = None
        model.zi = 0

class simple_QP(_ModelClassBase):
    """
    A continuous model with a quadratic objective and linear constraints
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.quadratic_objective = True
        self.results_file = join(thisDir,"simple_QP.results")

    def descrStr(self):
        return "simple_QP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.a = Param(initialize=1.0)
        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=NonNegativeReals)

        model.inactive_obj = Objective(expr=model.y)
        model.inactive_obj.deactivate()
        model.obj = Objective(expr=model.x**2 + 3.0*model.inactive_obj**2 + 1.0)
        model.c1 = Constraint(expr=model.a <= model.y)
        model.c2 = Constraint(expr=2.0 <= model.x/model.a - model.y <= 10)

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x = 1
        model.y = 1

class simple_QP_nosuffixes(simple_QP):
    def __init__(self):
        simple_QP.__init__(self)
        self.disable_suffix_tests = True

class simple_MIQP(_ModelClassBase):
    """
    A mixed-integer model with a quadratic objective and linear constraints
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.integer = True
        self.quadratic_objective = True
        self.results_file = join(thisDir,"simple_MIQP.results")

    def descrStr(self):
        return "simple_MIQP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.a = Param(initialize=1.0)
        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=Binary)

        model.obj = Objective(expr=model.x**2 + 3.0*model.y**2)
        model.c1 = Constraint(expr=model.a <= model.y)
        model.c2 = Constraint(expr=2.0 <= model.x/model.a - model.y <= 10)

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x = 1
        model.y = 1

class simple_QCP(_ModelClassBase):
    """
    A continuous model with a quadratic objective and quadratics constraints
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.quadratic_objective = True
        self.quadratic_constraint = True
        self.results_file = join(thisDir,"simple_QCP.results")

    def descrStr(self):
        return "simple_QCP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=NonNegativeReals)
        model.z = Var(within=NonNegativeReals)
        model.fixed_var = Var()
        model.fixed_var.fix(0.2)
        model.q1 = Var(bounds=(None, 0.2))
        model.q2 = Var(bounds=(-2,None))
        model.obj = Objective(expr=model.x+model.q1-model.q2,sense=maximize)
        model.c0 = Constraint(expr=model.x+model.y+model.z == 1)
        model.qc0 = Constraint(expr=model.x**2 + model.y**2 + model.fixed_var <= model.z**2)
        model.qc1 = Constraint(expr=model.x**2 <= model.y*model.z)
        model.c = ConstraintList()
        model.c.add((0, -model.q1**2 + model.fixed_var, None))
        model.c.add((None, model.q2**2 + model.fixed_var, 5))

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x = 1
        model.y = 1
        model.z = 1

class simple_QCP_nosuffixes(simple_QCP):
    def __init__(self):
        simple_QCP.__init__(self)
        self.disable_suffix_tests = True

class simple_MIQCP(_ModelClassBase):
    """
    A mixed-integer model with a quadratic objective and quadratic constraints
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.integer = True
        self.quadratic_constraint = True
        self.results_file = join(thisDir,"simple_MIQCP.results")

    def descrStr(self):
        return "simple_MIQCP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.x = Var(within=Binary)
        model.y = Var(within=Binary)
        model.z = Var(within=Binary)

        model.obj = Objective(expr=model.x,sense=maximize)
        model.c0 = Constraint(expr=model.x+model.y+model.z == 1)
        model.qc0 = Constraint(expr=model.x**2 + model.y**2 <= model.z**2)
        model.qc1 = Constraint(expr=model.x**2 <= model.y*model.z)

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x = None
        model.y = None
        model.z = None

class simple_SOS1(_ModelClassBase):
    """
    A discrete linear model with sos1 constraints
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.integer = True
        self.sos1 = True
        self.results_file = join(thisDir,"simple_SOS1.results")

    def descrStr(self):
        return "simple_SOS1"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.a = Param(initialize=0.1)
        model.x = Var(within=NonNegativeReals)
        model.y = Var([1,2],within=NonNegativeReals)

        model.obj = Objective(expr=model.x + model.y[1]+2*model.y[2])
        model.c1 = Constraint(expr=model.a <= model.y[2])
        model.c2 = Constraint(expr=2.0 <= model.x <= 10.0)
        model.c3 = SOSConstraint(var=model.y, index=[1,2], sos=1)
        model.c4 = Constraint(expr=summation(model.y) == 1)

        # Make an empty SOSConstraint
        model.c5 = SOSConstraint(var=model.y, index=[1,2], sos=1)
        model.c5.set_items([],[])
        assert len(list(model.c5.get_items())) == 0

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.x = 0
        model.y[1] = 1
        model.y[2] = None

class simple_SOS2(_ModelClassBase):
    """
    A discrete linear model with sos2 constraints
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.integer = True
        self.sos2 = True
        self.results_file = join(thisDir,"simple_SOS2.results")

    def descrStr(self):
        return "simple_SOS2"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.f = Var()
        model.x = Var(bounds=(1,3))
        model.fi = Param([1,2,3],mutable=True)
        model.fi[1] = 1.0
        model.fi[2] = 2.0
        model.fi[3] = 0.0
        model.xi = Param([1,2,3],mutable=True)
        model.xi[1] = 1.0
        model.xi[2] = 2.0
        model.xi[3] = 3.0
        model.p = Var(within=NonNegativeReals)
        model.n = Var(within=NonNegativeReals)
        model.lmbda = Var([1,2,3])
        model.obj = Objective(expr=model.p+model.n)
        model.c1 = ConstraintList()
        model.c1.add(0.0 <= model.lmbda[1] <= 1.0)
        model.c1.add(0.0 <= model.lmbda[2] <= 1.0)
        model.c1.add(0.0 <= model.lmbda[3])
        model.c2 = SOSConstraint(var=model.lmbda, index=[1,2,3], sos=2)
        model.c3 = Constraint(expr=summation(model.lmbda) == 1)
        model.c4 = Constraint(expr=model.f==summation(model.fi,model.lmbda))
        model.c5 = Constraint(expr=model.x==summation(model.xi,model.lmbda))
        model.x = 2.75
        model.x.fixed = True

        # Make an empty SOSConstraint
        model.c6 = SOSConstraint(var=model.lmbda, index=[1,2,3], sos=2)
        model.c6.set_items([],[])
        assert len(list(model.c6.get_items())) == 0

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        model.f = 0
        model.x = 2.75 # Fixed
        model.p = 1
        model.n = 0
        model.lmbda[1] = None
        model.lmbda[2] = None
        model.lmbda[3] = 1

class duals_maximize(_ModelClassBase):
    """
    A continuous linear model designed to test every form of
    constraint when collecting duals for a maximization
    objective
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.results_file = join(thisDir,"duals_maximize.results")

    def descrStr(self):
        return "duals_maximize"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.neg1 = Param(initialize=-1.0, mutable=True)
        model.pos1 = Param(initialize=1.0, mutable=True)

        model.s = RangeSet(1,12)
        model.x = Var(model.s)
        model.x[1].setlb(model.neg1)
        model.x[1].setub(model.pos1)
        model.x[2].setlb(model.neg1)
        model.x[2].setub(model.pos1)
        model.obj = Objective(expr=sum(model.x[i]*((-1)**(i))
                                       for i in model.x.index_set()),
                              sense=maximize)
        model.c = ConstraintList()
        # to make the variable used in the constraint match the name
        model.c.add(Constraint.Skip)
        model.c.add(Constraint.Skip)
        model.c.add(model.x[3]>=-1.)
        model.c.add(model.x[4]<=1.)
        model.c.add(model.x[5]==-1.)
        model.c.add(model.x[6]==-1.)
        model.c.add(model.x[7]==1.)
        model.c.add(model.x[8]==1.)
        model.c.add((model.neg1,model.x[9],model.neg1))
        model.c.add((-1.,model.x[10],-1.))
        model.c.add((1.,model.x[11],1.))
        model.c.add((1.,model.x[12],1.))

        model.c_inactive = ConstraintList()
        # to make the variable used in the constraint match the name
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(model.x[3]>=-2.)
        model.c_inactive.add(model.x[4]<=2.)

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        for i in model.s:
            model.x[i] = None

class duals_minimize(_ModelClassBase):
    """
    A continuous linear model designed to test every form of
    constraint when collecting duals for a minimization
    objective
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.results_file = join(thisDir,"duals_minimize.results")

    def descrStr(self):
        return "duals_minimize"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.s = RangeSet(1,12)
        model.x = Var(model.s)
        model.x[1].setlb(-1)
        model.x[1].setub(1)
        model.x[2].setlb(-1)
        model.x[2].setub(1)
        model.obj = Objective(expr=sum(model.x[i]*((-1)**(i+1))
                                       for i in model.x.index_set()))
        model.c = ConstraintList()
        # to make the variable used in the constraint match the name
        model.c.add(Constraint.Skip)
        model.c.add(Constraint.Skip)
        model.c.add(model.x[3]>=-1.)
        model.c.add(model.x[4]<=1.)
        model.c.add(model.x[5]==-1.)
        model.c.add(model.x[6]==-1.)
        model.c.add(model.x[7]==1.)
        model.c.add(model.x[8]==1.)
        model.c.add((-1.,model.x[9],-1.))
        model.c.add((-1.,model.x[10],-1.))
        model.c.add((1.,model.x[11],1.))
        model.c.add((1.,model.x[12],1.))

        model.c_inactive = ConstraintList()
        # to make the variable used in the constraint match the name
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(model.x[3]>=-2.)
        model.c_inactive.add(model.x[4]<=2.)

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        for i in model.s:
            model.x[i] = None

class compiled_LP(_ModelClassBase):
    """
    A continuous linear model that is compiled into a
    MatrixConstraint object.
    """
    def __init__(self):
        _ModelClassBase.__init__(self)
        self.linear = True
        self.results_file = join(thisDir,"compiled_LP.results")
        self.disable_suffix_tests = True

    def descrStr(self):
        return "compiled_LP"

    def generateModel(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model.name = self.descrStr()

        model.s = RangeSet(1,12)
        model.x = Var(model.s)
        model.x[1].setlb(-1)
        model.x[1].setub(1)
        model.x[2].setlb(-1)
        model.x[2].setub(1)
        model.obj = Objective(expr=sum(model.x[i]*((-1)**(i+1))
                                       for i in model.x.index_set()))
        model.c = ConstraintList()
        # to make the variable used in the constraint match the name
        model.c.add(Constraint.Skip)
        model.c.add(Constraint.Skip)
        model.c.add(model.x[3]>=-1.)
        model.c.add(model.x[4]<=1.)
        model.c.add(model.x[5]==-1.)
        model.c.add(model.x[6]==-1.)
        model.c.add(model.x[7]==1.)
        model.c.add(model.x[8]==1.)
        model.c.add((-1.,model.x[9],-1.))
        model.c.add((-1.,model.x[10],-1.))
        model.c.add((1.,model.x[11],1.))
        model.c.add((1.,model.x[12],1.))
        cdata = model.c.add((0, 1, 3))
        assert cdata.lower == 0
        assert cdata.upper == 3
        assert cdata.body() == 1
        assert not cdata.equality
        cdata = model.c.add((0, 2, 3))
        assert cdata.lower == 0
        assert cdata.upper == 3
        assert cdata.body() == 2
        assert not cdata.equality
        cdata = model.c.add((0, 1, None))
        assert cdata.lower == 0
        assert cdata.upper is None
        assert cdata.body() == 1
        assert not cdata.equality
        cdata = model.c.add((None, 0, 1))
        assert cdata.lower is None
        assert cdata.upper == 1
        assert cdata.body() == 0
        assert not cdata.equality
        cdata = model.c.add((1,1))
        assert cdata.lower == 1
        assert cdata.upper == 1
        assert cdata.body() == 1
        assert cdata.equality

        model.fixed_var = Var()
        model.fixed_var.fix(1.0)
        cdata = model.c.add((0, 1+model.fixed_var, 3))
        cdata = model.c.add((0, 2 + model.fixed_var, 3))
        cdata = model.c.add((0, model.fixed_var, None))
        cdata = model.c.add((None, model.fixed_var, 1))
        cdata = model.c.add((model.fixed_var, 1))

        model.c_inactive = ConstraintList()
        # to make the variable used in the constraint match the name
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(model.x[3]>=-2.)
        model.c_inactive.add(model.x[4]<=2.)

        compile_block_linear_constraints(model, 'Amatrix')

    def warmstartModel(self):
        assert self.model is not None
        model = self.model
        for i in model.s:
            model.x[i] = None

if __name__ == "__main__":
    import pyomo.environ
    from pyomo.opt import *
    #M = piecewise_LP()
    #M = compiled_LP()
    M = trivial_constraints_LP()
    #M = discrete_var_bounds_MILP()
    #M = simple_QCP()
    M.generateModel()
    M.warmstartModel()
    model = M.model
    model.pprint()
    #model.write("junk.mps", format='mps', io_options={'symbolic_solver_labels':True})
    #model.write("junk.lp", format='lp', io_options={'symbolic_solver_labels':True})
    #exit()
    #model.iis = Suffix(direction=Suffix.IMPORT)
    #model.dual = Suffix(direction=Suffix.IMPORT)
    #model.rc = Suffix(direction=Suffix.IMPORT)
    #model.slack = Suffix(direction=Suffix.IMPORT)
    #model.rc = Suffix(direction=Suffix.IMPORT)
    #model.dual = Suffix(direction=Suffix.IMPORT)

    #model.preprocess()
    #for block in model.block_data_objects(active=True):
    #    print(block.cname(True))
    #    block._canonical_repn.pprint()

    #model.write(format=None,filename="junk.nl",symbolic_solver_labels=True)
    #model.pprint()

    opt = SolverFactory("cbc", solver_io="mps")
    #opt = SolverFactory("gurobi", solver_io='mps')
    #opt = SolverFactory("glpk", solver_io='mps')
    #opt = SolverFactory("pico", solver_io='nl')
    #opt = SolverFactory("cplex", solver_io='python')
    #opt = SolverFactory("gurobi_ampl")
    #opt = SolverFactory("baron")
    #opt.options['NumLoc'] = 10
    #opt.options['preprocessing_presolve'] = False
    #opt = SolverFactory("cplexamp")
    #opt = SolverFactory("pico", solver_io="nl")

    #opt.options['write'] = 'infeas.iis'
    #model.cccc = Constraint(expr=model.x <= -1)

    results = opt.solve(model,
                        keepfiles=True,
#                        symbolic_solver_labels=True,
                        tee=True)
#                        warmstart=True,
#                        load_solutions=False)

    #print(results)
    #model.solutions.load_from(results)
    #model.dual.pprint(verbose=True)
    #model.rc.pprint(verbose=True)
    #model.slack.pprint(verbose=True)
    #model.pprint(verbose=True)
    #M.saveCurrentSolution("junk")#,suffixes=['dual','rc','slack'])
    #print(M.validateCurrentSolution(suffixes=['dual','rc','slack']))

    """
    opt = SolverFactory("cplex")

    import glob
    import json
    flist = glob.glob('*.results')
    print len(flist)


    for name in flist:
        M = eval(name.split('.')[0]+"()")
        M.generateModel()
        model = M.model
        model.load(opt.solve(model))
        print
        print name
        with open(name) as f:
            results = json.load(f)
        for block in model.block_data_objects():
            for var in components_data(block,Var):
                if 'stale' not in results[var.cname(True)]:
                    results[var.cname(True)]['stale'] = var.stale
                print '\t%18s'%var.cname(True), results[var.cname(True)]
        with open(name,'w') as f:
            json.dump(results, f, indent=2)
    """
