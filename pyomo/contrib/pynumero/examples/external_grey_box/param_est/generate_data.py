import pyomo.environ as pyo
import numpy.random as rnd
import pyomo.contrib.pynumero.examples.external_grey_box.param_est.models as pm

def generate_data(N, UA_mean, UA_std):
    m = pyo.ConcreteModel()
    pm.build_single_point_model_pyomo_only(m)

    # dummy objective since this is a square problem
    m.obj = pyo.Objective(expr=1) 

    # create the ipopt solver
    solver = pyo.SolverFactory('ipopt')

    print('run, Th_in, Tc_in, Th_out, Tc_out, UA')
    for i in range(N):
        # draw a random value for the parameters
        ua = float(rnd.normal(UA_mean, UA_std))
        # draw a noisy value for the test input conditions
        Th_in = 100 + float(rnd.normal(0, 2))
        Tc_in = 30  + float(rnd.normal(0, 2))
        m.UA.fix(ua)
        m.Th_in.fix(Th_in)
        m.Tc_in.fix(Tc_in)

        #solver.options['halt_on_ampl_error'] = 'yes'
        status = solver.solve(m, tee=True)
        print('{}, {}, {}, {}, {}, {}'.format(
            i,
            pyo.value(m.Th_in),
            pyo.value(m.Tc_in),
            pyo.value(m.Th_out),
            pyo.value(m.Tc_out),
            pyo.value(m.UA))
        )

def generate_data_external(N, UA_mean, UA_std):
    m = pyo.ConcreteModel()
    pm.build_single_point_model_external(m)

    # ADD MUTABLE PARAMETERS HERE FOR EQUALITIES
    m.UA_spec = pyo.Param(initialize=200, mutable=True)
    m.Th_in_spec = pyo.Param(initialize=100, mutable=True)
    m.Tc_in_spec = pyo.Param(initialize=30, mutable=True)
    m.UA_spec_con = pyo.Constraint(expr=m.egb.inputs['UA'] == m.UA_spec)
    m.Th_in_spec_con = pyo.Constraint(expr=m.egb.inputs['Th_in'] == m.Th_in_spec)
    m.Tc_in_spec_con = pyo.Constraint(expr=m.egb.inputs['Tc_in'] == m.Tc_in_spec)

    
    # dummy objective since this is a square problem
    m.obj = pyo.Objective(expr=(m.egb.inputs['UA'] - m.UA_spec)**2)

    # create the ipopt solver
    solver = pyo.SolverFactory('cyipopt')

    print('run, Th_in, Tc_in, Th_out, Tc_out, UA')
    for i in range(N):
        # draw a random value for the parameters
        UA = float(rnd.normal(UA_mean, UA_std))
        # draw a noisy value for the test input conditions
        Th_in = 100 + float(rnd.normal(0, 2))
        Tc_in = 30  + float(rnd.normal(0, 2))
        m.UA_spec.value = UA
        m.Th_in_spec.value = Th_in
        m.Tc_in_spec.value = Tc_in

        #solver.options['halt_on_ampl_error'] = 'yes'
        status = solver.solve(m, tee=True)
        print('{}, {}, {}, {}, {}, {}'.format(
            i,
            pyo.value(m.egb.inputs['Th_in']),
            pyo.value(m.egb.inputs['Tc_in']),
            pyo.value(m.egb.inputs['Th_out']),
            pyo.value(m.egb.inputs['Tc_out']),
            pyo.value(m.egb.inputs['UA']))
        )

if __name__ == '__main__':
    rnd.seed(42)
    generate_data(50, 200, 5)
    #generate_data_external(1, 200, 5)

    
