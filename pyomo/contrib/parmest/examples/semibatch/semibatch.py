# This file contains the semibatch model and callback. Data is loaded from 
# files in the callback.

import re
import json
from pyomo.environ import *
from pyomo.dae import *

def generate_model(data):

    # unpack and fix the data
    cameastemp = data['Ca_meas']
    cbmeastemp = data['Cb_meas']
    ccmeastemp = data['Cc_meas']
    trmeastemp = data['Tr_meas']

    cameas={}
    cbmeas={}
    ccmeas={}
    trmeas={}
    for i in cameastemp.keys():
        cameas[float(i)] = cameastemp[i]
        cbmeas[float(i)] = cbmeastemp[i]
        ccmeas[float(i)] = ccmeastemp[i]
        trmeas[float(i)] = trmeastemp[i]

    m = ConcreteModel()

    #
    # Measurement Data
    #
    m.measT = Set(initialize=sorted(cameas.keys()))
    m.Ca_meas = Param(m.measT, initialize=cameas)
    m.Cb_meas = Param(m.measT, initialize=cbmeas)
    m.Cc_meas = Param(m.measT, initialize=ccmeas)
    m.Tr_meas = Param(m.measT, initialize=trmeas)

    #
    # Parameters for semi-batch reactor model
    #
    m.R = Param(initialize=8.314) # kJ/kmol/K
    m.Mwa = Param(initialize=50.0) # kg/kmol
    m.rhor = Param(initialize=1000.0) # kg/m^3
    m.cpr = Param(initialize=3.9) # kJ/kg/K
    m.Tf = Param(initialize=300) # K
    m.deltaH1 = Param(initialize=-40000.0) # kJ/kmol
    m.deltaH2 = Param(initialize=-50000.0) # kJ/kmol
    m.alphaj = Param(initialize=0.8) # kJ/s/m^2/K
    m.alphac = Param(initialize=0.7) # kJ/s/m^2/K
    m.Aj = Param(initialize=5.0) # m^2
    m.Ac = Param(initialize=3.0) # m^2
    m.Vj = Param(initialize=0.9) # m^3
    m.Vc = Param(initialize=0.07) # m^3
    m.rhow = Param(initialize=700.0) # kg/m^3
    m.cpw = Param(initialize=3.1) # kJ/kg/K
    m.Ca0 = Param(initialize=data['Ca0']) # kmol/m^3)
    m.Cb0 = Param(initialize=data['Cb0']) # kmol/m^3)
    m.Cc0 = Param(initialize=data['Cc0']) # kmol/m^3)
    m.Tr0 = Param(initialize=300.0) # K
    m.Vr0 = Param(initialize=1.0) # m^3

    m.time = ContinuousSet(bounds=(0, 21600), initialize=m.measT)  # Time in seconds

    #
    # Control Inputs
    #
    def _initTc(m, t):
        if t < 10800:
            return data['Tc1']
        else:
            return data['Tc2']
    m.Tc = Param(m.time, initialize=_initTc, default=_initTc)  # bounds= (288,432) Cooling coil temp, control input

    def _initFa(m, t):
        if t < 10800:
            return data['Fa1']
        else:
            return data['Fa2']
    m.Fa = Param(m.time, initialize=_initFa, default=_initFa)  # bounds=(0,0.05) Inlet flow rate, control input

    #
    # Parameters being estimated
    #
    m.k1 = Var(initialize=14, bounds=(2,100))  # 1/s Actual: 15.01
    m.k2 = Var(initialize=90, bounds=(2,150))  # 1/s Actual: 85.01
    m.E1 = Var(initialize=27000.0, bounds=(25000,40000))  # kJ/kmol Actual: 30000
    m.E2 = Var(initialize=45000.0, bounds=(35000,50000))  # kJ/kmol Actual: 40000
    # m.E1.fix(30000)
    # m.E2.fix(40000)


    #
    # Time dependent variables
    #
    m.Ca = Var(m.time, initialize=m.Ca0, bounds=(0,25))
    m.Cb = Var(m.time, initialize=m.Cb0, bounds=(0,25))
    m.Cc = Var(m.time, initialize=m.Cc0, bounds=(0,25))
    m.Vr = Var(m.time, initialize=m.Vr0)
    m.Tr = Var(m.time, initialize=m.Tr0)
    m.Tj = Var(m.time, initialize=310.0, bounds=(288,None)) # Cooling jacket temp, follows coil temp until failure

    #
    # Derivatives in the model
    #
    m.dCa = DerivativeVar(m.Ca)
    m.dCb = DerivativeVar(m.Cb)
    m.dCc = DerivativeVar(m.Cc)
    m.dVr = DerivativeVar(m.Vr)
    m.dTr = DerivativeVar(m.Tr)

    #
    # Differential Equations in the model
    #

    def _dCacon(m,t):
        if t == 0:
            return Constraint.Skip
        return m.dCa[t] == m.Fa[t]/m.Vr[t] - m.k1*exp(-m.E1/(m.R*m.Tr[t]))*m.Ca[t]
    m.dCacon = Constraint(m.time, rule=_dCacon)

    def _dCbcon(m,t):
        if t == 0:
            return Constraint.Skip
        return m.dCb[t] == m.k1*exp(-m.E1/(m.R*m.Tr[t]))*m.Ca[t] - \
                           m.k2*exp(-m.E2/(m.R*m.Tr[t]))*m.Cb[t]
    m.dCbcon = Constraint(m.time, rule=_dCbcon)

    def _dCccon(m,t):
        if t == 0:
            return Constraint.Skip
        return m.dCc[t] == m.k2*exp(-m.E2/(m.R*m.Tr[t]))*m.Cb[t]
    m.dCccon = Constraint(m.time, rule=_dCccon)

    def _dVrcon(m,t):
        if t == 0:
            return Constraint.Skip
        return m.dVr[t] == m.Fa[t]*m.Mwa/m.rhor
    m.dVrcon = Constraint(m.time, rule=_dVrcon)

    def _dTrcon(m,t):
        if t == 0:
            return Constraint.Skip
        return m.rhor*m.cpr*m.dTr[t] == \
               m.Fa[t]*m.Mwa*m.cpr/m.Vr[t]*(m.Tf-m.Tr[t]) - \
               m.k1*exp(-m.E1/(m.R*m.Tr[t]))*m.Ca[t]*m.deltaH1 - \
               m.k2*exp(-m.E2/(m.R*m.Tr[t]))*m.Cb[t]*m.deltaH2 + \
               m.alphaj*m.Aj/m.Vr0*(m.Tj[t]-m.Tr[t]) + \
               m.alphac*m.Ac/m.Vr0*(m.Tc[t]-m.Tr[t])
    m.dTrcon = Constraint(m.time, rule=_dTrcon)

    def _singlecooling(m,t):
        return m.Tc[t] == m.Tj[t]
    m.singlecooling = Constraint(m.time, rule=_singlecooling)

    # Initial Conditions
    def _initcon(m):
        yield m.Ca[m.time.first()] == m.Ca0
        yield m.Cb[m.time.first()] == m.Cb0
        yield m.Cc[m.time.first()] == m.Cc0
        yield m.Vr[m.time.first()] == m.Vr0
        yield m.Tr[m.time.first()] == m.Tr0
    m.initcon = ConstraintList(rule=_initcon)

    #
    # Stage-specific cost computations
    #
    def ComputeFirstStageCost_rule(model):
        return 0
    m.FirstStageCost = Expression(rule=ComputeFirstStageCost_rule)

    def AllMeasurements(m):
        return sum((m.Ca[t] - m.Ca_meas[t]) ** 2 + (m.Cb[t] - m.Cb_meas[t]) ** 2 
                   + (m.Cc[t] - m.Cc_meas[t]) ** 2
                   + 0.01 * (m.Tr[t] - m.Tr_meas[t]) ** 2 for t in m.measT)

    def MissingMeasurements(m):
        if data['experiment'] == 1:
            return sum((m.Ca[t] - m.Ca_meas[t]) ** 2 + (m.Cb[t] - m.Cb_meas[t]) ** 2 
                       + (m.Cc[t] - m.Cc_meas[t]) ** 2
                       + (m.Tr[t] - m.Tr_meas[t]) ** 2 for t in m.measT)
        elif data['experiment'] == 2:
            return sum((m.Tr[t] - m.Tr_meas[t]) ** 2 for t in m.measT)
        else:
            return sum((m.Cb[t] - m.Cb_meas[t]) ** 2 
                       + (m.Tr[t] - m.Tr_meas[t]) ** 2 for t in m.measT)

    m.SecondStageCost = Expression(rule=MissingMeasurements)

    def total_cost_rule(model):
        return model.FirstStageCost + model.SecondStageCost
    m.Total_Cost_Objective = Objective(rule=total_cost_rule, sense=minimize)

    # Discretize model
    disc = TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=20, ncp=4)
    return m

# Number of scenarios
scenarios = 10

def pysp_scenario_tree_model_callback():
    from pyomo.pysp.scenariotree.tree_structure_model \
        import CreateConcreteTwoStageScenarioTreeModel

    st_model = CreateConcreteTwoStageScenarioTreeModel(scenarios)

    first_stage = st_model.Stages.first()
    second_stage = st_model.Stages.last()

    # First Stage
    st_model.StageCost[first_stage] = 'FirstStageCost'
    st_model.StageVariables[first_stage].add('k1')
    st_model.StageVariables[first_stage].add('k2')
    st_model.StageVariables[first_stage].add('E1')
    st_model.StageVariables[first_stage].add('E2')

    # Second Stage
    st_model.StageCost[second_stage] = 'SecondStageCost'
    # st_model.StageVariables[second_stage].add('Ca[*]')
    # st_model.StageVariables[second_stage].add('Cb[*]')
    # st_model.StageVariables[second_stage].add('Cc[*]')
    # st_model.StageVariables[second_stage].add('Tr[*]')
    #st_model.StageVariables[second_stage].add('QuantitySuperQuotaSold[*]')
    #st_model.StageVariables[second_stage].add('QuantityPurchased[*]')

    return st_model


def pysp_instance_creation_callback(scenario_tree_model,
                                    scenario_name,
                                    node_names):

    exp_num = int(re.compile(r'(\d+)$').search(scenario_name).group(1))
    """
    # Experiments with measurement noise
    # exp_num = [1,2,3] # Different step changes in control inputs
    # exp_num = [4,5,6] # Different concentration initial conditions

    # Experiments with errors in model parameters
    # exp_num = [7, 9, 11, 13]  # No noise
    # exp_num = [8, 10, 12, 14]  # With noise
    """
    fname = 'exp'+str(exp_num)+'.out'
    with open(fname,'r') as infile:
        data = json.load(infile)
    data['experiment'] = exp_num
        
    model = generate_model(data)

    # see about fixing theta
    """ now done in parmest
    if hasattr(scenario_tree_model, "ThetaVals"):
        thetavals = scenario_tree_model.ThetaVals

        # TBD Dec 2017, dlw: indexed theta
        for vstr in thetavals:
            if thetavals[vstr] is not None:
                #print ("Fixing",vstr,"at",str(thetavals[vstr]))
                getattr(model, vstr).fix(thetavals[vstr])
            else:
                #print ("Freeing",vstr)
                getattr(model, vstr).fixed = False
    """
    
    return model

if __name__ == '__main__':

    model = pysp_instance_creation_callback(None, 'foo2', None)
    solver = SolverFactory('ipopt')
    solver.solve(model)
    model.display()
    model.pprint()
    print('k1 = ', model.k1())
    print('E1 = ', model.E1())
    
