import pyomo.environ as pyo
from pyomo.common.fileutils import find_library

if __name__ == "__main__":
    lib = find_library("cspline_1d_external")
    params = "t1_params.txt"

    m = pyo.ConcreteModel()
    m.f = pyo.ExternalFunction(library=lib, function="cspline_1d")

    m.x = pyo.Var(initialize=2)  # , bounds=(0.9, 5.1))
    m.y = pyo.Var()

    m.c1 = pyo.Constraint(expr=m.y == m.f(m.x, params))

    m.o = pyo.Objective(expr=-m.y)

    print(pyo.value(m.f(0.9, params)))
    print(pyo.value(m.f(1.01, params)))
    print(pyo.value(m.f(1.05, params)))
    print(pyo.value(m.f(1.2, params)))
    print(pyo.value(m.f(1.3, params)))
    print(pyo.value(m.f(1.5, params)))
    print(pyo.value(m.f(2, params)))
    print(pyo.value(m.f(2.01, params)))
    print(pyo.value(m.f(2.1, params)))
    print(pyo.value(m.f(3.0, params)))
    print(pyo.value(m.f(3.1, params)))
    print(pyo.value(m.f(4.9, params)))
    print(pyo.value(m.f(5.0, params)))
    print(pyo.value(m.f(5.1, params)))

    import idaes

    solver_obj = pyo.SolverFactory("ipopt")
    solver_obj.solve(m, tee=True)

    m.display()
