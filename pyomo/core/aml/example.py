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
# AbstractModel
#

m = Block()
assert not m.constructed
assert m.ctype is core_old.Block
assert isinstance(m, Block)
assert isinstance(m, _AddToConstructBlock)
assert isinstance(m, SimpleBlock)
assert isinstance(m, pk.block)

m.x = Var()
assert not m.x.constructed
assert m.x.ctype is core_old.Var
assert isinstance(m.x, Var)
assert isinstance(m.x, SimpleVar)
assert isinstance(m.x, pk.variable)

m.X = Var([1,2,3])
assert not m.X.constructed
assert m.X.ctype is core_old.Var
assert len(m.X) == 0
assert isinstance(m.x, Var)
assert isinstance(m.X, IndexedVar)
assert isinstance(m.X, pk.variable_dict)

m.b = Block()
assert not m.b.constructed
assert m.b.ctype is core_old.Block
assert isinstance(m.b, Block)
assert isinstance(m.b, _AddToConstructBlock)
assert isinstance(m.b, SimpleBlock)
assert isinstance(m.b, pk.block)

def B_rule(b):
    pass
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
assert m.X.constructed
assert len(m.X) == 3
for i in m.X:
    assert m.X[i].ctype is core_old.Var
    assert isinstance(m.X[i], pk.variable)
assert m.b.constructed
assert m.B.constructed
assert len(m.B) == 3
for i in m.B:
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

m.x = Var()
assert m.x.constructed
assert m.x.ctype is core_old.Var
assert isinstance(m.x, Var)
assert isinstance(m.x, SimpleVar)
assert isinstance(m.x, pk.variable)

m.X = Var([1,2,3])
assert m.X.constructed
assert len(m.X) == 3
assert m.X.ctype is core_old.Var
assert isinstance(m.X, Var)
assert isinstance(m.X, IndexedVar)
assert isinstance(m.X, pk.variable_dict)
for i in m.X:
    assert m.X[i].ctype is core_old.Var
    assert isinstance(m.X[i], pk.variable)

m.b = Block()
assert m.b.constructed
assert m.b.ctype is core_old.Block
assert isinstance(m.b, Block)
assert isinstance(m.b, _AddToConstructBlock)
assert isinstance(m.b, SimpleBlock)
assert isinstance(m.b, pk.block)

def B_rule(b):
    pass
m.B = Block([1,2,3], rule=B_rule)
assert m.B.constructed
assert len(m.B) == 3
assert m.B.ctype is core_old.Block
assert isinstance(m.B, Block)
assert isinstance(m.B, IndexedBlock)
assert isinstance(m.B, pk.block_dict)
for i in m.B:
    assert m.B[i].constructed
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
