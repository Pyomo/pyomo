# Edge case based on a user's project
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import pandas as pd

from pyomo.environ import *
import pyomo.contrib.parmest.parmest as parmest


def model_function(data):
    model = ConcreteModel()
    model.T = Set(initialize=data['global']['T'], ordered=True)
    model.J = Set(initialize=[1])

    def boiler_rule(A, u):

        A.eta = Var(bounds=(0, 1), initialize=data['Boiler'][u]['eta'][None])
        A.eta.fix()
        A.cost_per_kWh = Param(initialize=1)

        A.P_in = Var(model.T, within=NonNegativeReals)
        A.P_out = Var(model.T, within=NonNegativeReals)

        # constraint efficiancy
        def eta_constraint_rule(A, t):
            return A.P_out[t] == A.P_in[t] * A.eta

        A.constraint_eta = Constraint(model.T, rule=eta_constraint_rule)

    model.Boiler = Block(model.J, rule=boiler_rule)
    model.Boiler.construct(data=data['Boiler'])

    model.demand = Param(model.T, initialize={0: 0, 1: 600, 2: 900, 3: 1200})
    model.cost_error = Param(initialize=100)

    def Bilanz_Regel(model, t):
        return model.demand[t] == model.Boiler[1].P_out[t]

    model.balance = Constraint(model.T, rule=Bilanz_Regel)

    model.Zielfunktion = Objective(expr=sum(
        model.Boiler[1].P_in[t] * model.Boiler[1].cost_per_kWh * model.cost_error for t in model.T))
    return model


def parameter_estimation():
    """Estimate Parameter Boiler[1].eta"""
    data = {'global': {'T': [0, 1, 2, 3]},
            'Boiler': {1: {'cost_per_kWh': {None: 1},
                           'eta': {None: 10}}},  # during construction eta will be fixed to wrong value.
            # paramest schould unfix the variable and estimate the correct value.
            'P_in': {0: 0.0, 1: 1200.0, 2: 1800.0, 3: 2400.0},
            'P_out': {0: 0.0, 1: 600.0, 2: 900.0, 3: 1200.0}}
    data = [data]

    # theta_names
    # The third argument is a list of variable names that the user wants to estimate.
    # The list contains strings with Var names from the Pyomo model.
    theta_names = ['Boiler[1].eta']

    # objective_function
    def objective_function(model, data):
        T = data['global']['T']
        expr = 0
        for t in T:
            # (observed_value - value_calculated_form_model)^2*1/nr_time_steps
            # Achtung: Geht davon aus, dass alle Zeitschritte gleich lang sind.
            # Muss für variable Schrittweite ggf. angepasst werden.
            # P_in
            expr += ((data['P_in'][t] - model.Boiler[1].P_in[t]) ** 2) * (1 / len(T))
            # P_out
            expr += ((data['P_in'][t] - model.Boiler[1].P_out[t]) ** 2) * (1 / len(T))
        return expr

    pest = parmest.Estimator(model_function, data, theta_names, objective_function)
    obj, theta = pest.theta_est()
    print("obj={}; theta={}".format(obj, theta))

def solve_model():
    """this represents a normal solve of the model, without parameter estimation."""

    data = {'global': {'T': [0, 1, 2, 3]},
            'Boiler': {1: {'eta':{None: 0.5},'cost_per_kWh': {None: 1}}}}


    instance = model_function(data)
    # solver
    result = SolverFactory('glpk').solve(instance, tee=True)
    print(result)

    # plotte Verbrauch, den die Anlagen erfüllen müssen
    plt.figure(1)
    plot(pd.Series(instance.demand.extract_values()), drawstyle='steps-post')
    plt.show();
    print('Demand')

    # plotte die bereitgestellte Leistung P_out der Anlagen 1(blau) und 2(orange)
    # der Verbrauch ist noch mal gepunktet dargestellt
    plt.figure(2)
    plot(pd.Series(instance.Boiler[1].P_out.extract_values()), drawstyle='steps-post')

    plot(pd.Series(instance.demand.extract_values()), 'k:', drawstyle='steps-post')
    plt.legend(['Boiler 1', 'demand'])
    plt.show();

    print(instance.Boiler[1].P_in.extract_values())
    print(instance.Boiler[1].P_out.extract_values())


#solve_model()
parameter_estimation()
