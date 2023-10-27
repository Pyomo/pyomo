import pyomo.environ as pe
import coramin
from coramin.utils.plot_relaxation import plot_relaxation
from pyomo.contrib import appsi


def main():
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(0.001, 2))
    m.y = pe.Var(bounds=(0.01, 10))
    m.z = pe.Var()
    m.c = coramin.relaxations.AlphaBBRelaxation()

    m.c.build(
        aux_var=m.z,
        f_x_expr=m.x*pe.log(m.x/m.y),
        relaxation_side=coramin.RelaxationSide.UNDER,
        eigenvalue_bounder=coramin.EigenValueBounder.GershgorinWithSimplification,
    )
    m.x.value = m.x.lb
    m.y.value = m.y.ub
    m.c.add_cut(keep_cut=True, check_violation=False)
    m.x.value = m.x.ub
    m.y.value = m.y.lb
    m.c.add_cut(keep_cut=True, check_violation=False)

    opt = pe.SolverFactory('gurobi_persistent')
    opt.set_instance(m)
    plot_relaxation(m, m.c, opt)

    m.c.hessian.method = coramin.EigenValueBounder.LinearProgram
    m.c.hessian.opt = appsi.solvers.Gurobi()
    m.c.rebuild()
    plot_relaxation(m, m.c, opt)

    m.c.hessian.method = coramin.EigenValueBounder.Global
    mip_opt = appsi.solvers.Gurobi()
    nlp_opt = appsi.solvers.Ipopt()
    eigenvalue_opt = coramin.algorithms.MultiTree(
        mip_solver=mip_opt,
        nlp_solver=nlp_opt,
    )
    eigenvalue_opt.config.convexity_effort = 'medium'
    m.c.hessian.opt = eigenvalue_opt
    m.c.rebuild()
    plot_relaxation(m, m.c, opt)


if __name__ == '__main__':
    main()
