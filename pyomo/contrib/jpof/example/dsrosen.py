import pyomo.environ as pe
import pyomo.contrib.jpof

def create(N):
    M = pe.ConcreteModel()

    def x_init(model, i):
        if i % 2 == 1:
            return -1.2 
        else:
            return 1.2

    M.x = pe.Var(pe.RangeSet(1,N), initialize=x_init)
    M.p = pe.Param(initialize=0, mutable=True)
    M.I = pe.RangeSet(1,N-1)

    M.f = pe.Objective(expr=sum( 100*(M.x[i+1]-M.x[i]**2+M.p)**2 + (1-M.x[i])**2 for i in M.I))

    return M

writer = pyomo.contrib.jpof.JPOFWriter()

M = create(3)
writer(M, filename="dsrosen3.jpof", io_options={'file_determinism':3, 'symbolic_solver_labels':True})
M = create(6)
writer(M, filename="dsrosen6.jpof", io_options={'file_determinism':3, 'symbolic_solver_labels':True})
M = create(10)
writer(M, filename="dsrosen10.jpof", io_options={'file_determinism':3, 'symbolic_solver_labels':True})

