import pyomo.environ
from pyomo.core.aml.block import (Block,
                                  SimpleBlock,
                                  IndexedBlock,
                                  _AddToConstructBlock)
from pyomo.core.aml.var import (Var,
                                SimpleVar,
                                IndexedVar)
import pyomo.core.kernel as pk
import pyomo.core.base as core_old

#
# Note: Right now, indexed objects only support a single set
# argument that needs to be something concrete (e.g., a set,
# list, tuple). Otherwise, testing this would have required
# reimplementing Set.
#

def domain_rule(m, *args):
    return pk.Integers
domain_init = pk.Integers
def bounds_rule(m, *args):
    return (0,1)
bounds_init = (0,1)
def value_rule(m, *args):
    return 2
value_init = 2

#
# AbstractModel
#

m = Block()
assert not m.constructed
assert m.ctype is core_old.Block
assert isinstance(m, Block)
assert isinstance(m, _AddToConstructBlock)
assert isinstance(m, SimpleBlock)
assert isinstance(m, pk.block)

m.x = Var(domain=domain_rule,
          bounds=bounds_rule,
          initialize=value_rule)
assert not m.x.constructed
assert m.x.ctype is core_old.Var
assert isinstance(m.x, Var)
assert isinstance(m.x, SimpleVar)
assert isinstance(m.x, pk.variable)

m.X = Var([1,2,3],
          domain=domain_rule,
          bounds=bounds_rule,
          initialize=value_rule)
assert not m.X.constructed
assert m.X.ctype is core_old.Var
assert len(m.X) == 0
assert isinstance(m.x, Var)
assert isinstance(m.X, IndexedVar)
assert isinstance(m.X, pk.variable_dict)

def b_rule(b):
    b.y = 2
m.b = Block(rule=b_rule)
assert not m.b.constructed
assert m.b.ctype is core_old.Block
assert isinstance(m.b, Block)
assert isinstance(m.b, _AddToConstructBlock)
assert isinstance(m.b, SimpleBlock)
assert isinstance(m.b, pk.block)

def B_rule(b):
    b.y = 2
m.B = Block([1,2,3], rule=B_rule)
assert not m.B.constructed
assert m.B.ctype is core_old.Block
assert len(m.B) == 0
assert isinstance(m.B, Block)
assert isinstance(m.B, IndexedBlock)
assert isinstance(m.B, pk.block_dict)

m.construct()
assert m.constructed
assert m.x.constructed
assert m.x.domain_type == pk.IntegerSet
assert m.x.bounds == (0,1)
assert m.x.value == 2
assert m.X.constructed
assert len(m.X) == 3
for i in m.X:
    assert m.X[i].domain_type == pk.IntegerSet
    assert m.X[i].bounds == (0,1)
    assert m.X[i].value == 2
    assert m.X[i].ctype is core_old.Var
    assert isinstance(m.X[i], pk.variable)
assert m.b.constructed
assert m.b.y == 2
assert m.B.constructed
assert len(m.B) == 3
for i in m.B:
    assert m.B[i].y == 2
    assert m.B[i].constructed
    assert m.B[i].ctype is core_old.Block
    assert isinstance(m.B[i], _AddToConstructBlock)
    assert isinstance(m.B[i], pk.block)

del m

#
# ConcreteModel
#

m = Block(concrete=True)
assert m.constructed
assert m.ctype is core_old.Block
assert isinstance(m, Block)
assert isinstance(m, _AddToConstructBlock)
assert isinstance(m, SimpleBlock)
assert isinstance(m, pk.block)

m.x = Var(domain=domain_init,
          bounds=bounds_init,
          initialize=value_init)
assert m.x.constructed
assert m.x.domain_type == pk.IntegerSet
assert m.x.bounds == (0,1)
assert m.x.value == 2
assert m.x.ctype is core_old.Var
assert isinstance(m.x, Var)
assert isinstance(m.x, SimpleVar)
assert isinstance(m.x, pk.variable)

m.X = Var([1,2,3],
          domain=domain_init,
          bounds=bounds_init,
          initialize=value_init)
assert m.X.constructed
assert len(m.X) == 3
assert m.X.ctype is core_old.Var
assert isinstance(m.X, Var)
assert isinstance(m.X, IndexedVar)
assert isinstance(m.X, pk.variable_dict)
for i in m.X:
    assert m.X[i].domain_type == pk.IntegerSet
    assert m.X[i].bounds == (0,1)
    assert m.X[i].value == 2
    assert m.X[i].ctype is core_old.Var
    assert isinstance(m.X[i], pk.variable)

def b_rule(b):
    b.y = 2
m.b = Block(rule=b_rule)
assert m.b.constructed
assert m.b.y == 2
assert m.b.ctype is core_old.Block
assert isinstance(m.b, Block)
assert isinstance(m.b, _AddToConstructBlock)
assert isinstance(m.b, SimpleBlock)
assert isinstance(m.b, pk.block)

def B_rule(b):
    b.y = 2
m.B = Block([1,2,3], rule=B_rule)
assert m.B.constructed
assert len(m.B) == 3
assert m.B.ctype is core_old.Block
assert isinstance(m.B, Block)
assert isinstance(m.B, IndexedBlock)
assert isinstance(m.B, pk.block_dict)
for i in m.B:
    assert m.B[i].constructed
    assert m.B[i].y == 2
    assert m.B[i].ctype is core_old.Block
    assert isinstance(m.B[i], _AddToConstructBlock)
    assert isinstance(m.B[i], pk.block)

cplex = pyomo.environ.SolverFactory("cplex")
# use newer base components to add an objective
# and a constraint
m.o = pk.objective(m.x)
m.c = pk.constraint(m.x >= 1)
status = cplex.solve(m)
assert str(status.solver.termination_condition) == 'optimal'
