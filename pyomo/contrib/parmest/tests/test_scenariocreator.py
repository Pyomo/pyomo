#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import pandas as pd, pandas_available

uuid_available = True
try:
    import uuid
except:
    uuid_available = False

import pyomo.common.unittest as unittest
import os
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.scenariocreator as sc
import pyomo.environ as pyo
from pyomo.environ import SolverFactory

ipopt_available = SolverFactory("ipopt").available()

testdir = os.path.dirname(os.path.abspath(__file__))


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestScenarioReactorDesign(unittest.TestCase):
    def setUp(self):
        from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
            ReactorDesignExperiment,
        )

        # Data from the design
        data = pd.DataFrame(
            data=[
                [1.05, 10000, 3458.4, 1060.8, 1683.9, 1898.5],
                [1.10, 10000, 3535.1, 1064.8, 1613.3, 1893.4],
                [1.15, 10000, 3609.1, 1067.8, 1547.5, 1887.8],
                [1.20, 10000, 3680.7, 1070.0, 1486.1, 1881.6],
                [1.25, 10000, 3750.0, 1071.4, 1428.6, 1875.0],
                [1.30, 10000, 3817.1, 1072.2, 1374.6, 1868.0],
                [1.35, 10000, 3882.2, 1072.4, 1324.0, 1860.7],
                [1.40, 10000, 3945.4, 1072.1, 1276.3, 1853.1],
                [1.45, 10000, 4006.7, 1071.3, 1231.4, 1845.3],
                [1.50, 10000, 4066.4, 1070.1, 1189.0, 1837.3],
                [1.55, 10000, 4124.4, 1068.5, 1148.9, 1829.1],
                [1.60, 10000, 4180.9, 1066.5, 1111.0, 1820.8],
                [1.65, 10000, 4235.9, 1064.3, 1075.0, 1812.4],
                [1.70, 10000, 4289.5, 1061.8, 1040.9, 1803.9],
                [1.75, 10000, 4341.8, 1059.0, 1008.5, 1795.3],
                [1.80, 10000, 4392.8, 1056.0, 977.7, 1786.7],
                [1.85, 10000, 4442.6, 1052.8, 948.4, 1778.1],
                [1.90, 10000, 4491.3, 1049.4, 920.5, 1769.4],
                [1.95, 10000, 4538.8, 1045.8, 893.9, 1760.8],
            ],
            columns=["sv", "caf", "ca", "cb", "cc", "cd"],
        )

        # Create an experiment list
        exp_list = []
        for i in range(data.shape[0]):
            exp_list.append(ReactorDesignExperiment(data, i))

        self.pest = parmest.Estimator(exp_list, obj_function='SSE')

    def test_scen_from_exps(self):
        scenmaker = sc.ScenarioCreator(self.pest, "ipopt")
        experimentscens = sc.ScenarioSet("Experiments")
        scenmaker.ScenariosFromExperiments(experimentscens)
        experimentscens.write_csv("delme_exp_csv.csv")
        df = pd.read_csv("delme_exp_csv.csv")
        os.remove("delme_exp_csv.csv")
        # March '20: all reactor_design experiments have the same theta values!
        k1val = df.loc[5].at["k1"]
        self.assertAlmostEqual(k1val, 5.0 / 6.0, places=2)
        tval = experimentscens.ScenarioNumber(0).ThetaVals["k1"]
        self.assertAlmostEqual(tval, 5.0 / 6.0, places=2)

    @unittest.skipIf(not uuid_available, "The uuid module is not available")
    def test_no_csv_if_empty(self):
        # low level test of scenario sets
        # verify that nothing is written, but no errors with empty set

        emptyset = sc.ScenarioSet("empty")
        tfile = uuid.uuid4().hex + ".csv"
        emptyset.write_csv(tfile)
        self.assertFalse(
            os.path.exists(tfile), "ScenarioSet wrote csv in spite of empty set"
        )


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestScenarioSemibatch(unittest.TestCase):
    def setUp(self):
        import pyomo.contrib.parmest.examples.semibatch.semibatch as sb
        import json

        self.fbase = os.path.join(testdir, "..", "examples", "semibatch")
        # Data, list of dictionaries
        data = []
        for exp_num in range(10):
            fname = "exp" + str(exp_num + 1) + ".out"
            fullname = os.path.join(self.fbase, fname)
            with open(fullname, "r") as infile:
                d = json.load(infile)
                data.append(d)

        # Note, the model already includes a 'SecondStageCost' expression
        # for the sum of squared error that will be used in parameter estimation

        # Create an experiment list
        exp_list = []
        for i in range(len(data)):
            exp_list.append(sb.SemiBatchExperiment(data[i]))

        self.pest = parmest.Estimator(exp_list)

    def test_semibatch_bootstrap(self):
        scenmaker = sc.ScenarioCreator(self.pest, "ipopt")
        bootscens = sc.ScenarioSet("Bootstrap")
        numtomake = 2
        scenmaker.ScenariosFromBootstrap(bootscens, numtomake, seed=1134)
        tval = bootscens.ScenarioNumber(0).ThetaVals["k1"]
        self.assertAlmostEqual(tval, 20.64, places=1)


###########################
# tests for deprecated UI #
###########################


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestScenarioReactorDesignDeprecated(unittest.TestCase):
    def setUp(self):

        def reactor_design_model(data):
            # Create the concrete model
            model = pyo.ConcreteModel()

            # Rate constants
            model.k1 = pyo.Param(
                initialize=5.0 / 6.0, within=pyo.PositiveReals, mutable=True
            )  # min^-1
            model.k2 = pyo.Param(
                initialize=5.0 / 3.0, within=pyo.PositiveReals, mutable=True
            )  # min^-1
            model.k3 = pyo.Param(
                initialize=1.0 / 6000.0, within=pyo.PositiveReals, mutable=True
            )  # m^3/(gmol min)

            # Inlet concentration of A, gmol/m^3
            if isinstance(data, dict) or isinstance(data, pd.Series):
                model.caf = pyo.Param(
                    initialize=float(data["caf"]), within=pyo.PositiveReals
                )
            elif isinstance(data, pd.DataFrame):
                model.caf = pyo.Param(
                    initialize=float(data.iloc[0]["caf"]), within=pyo.PositiveReals
                )
            else:
                raise ValueError("Unrecognized data type.")

            # Space velocity (flowrate/volume)
            if isinstance(data, dict) or isinstance(data, pd.Series):
                model.sv = pyo.Param(
                    initialize=float(data["sv"]), within=pyo.PositiveReals
                )
            elif isinstance(data, pd.DataFrame):
                model.sv = pyo.Param(
                    initialize=float(data.iloc[0]["sv"]), within=pyo.PositiveReals
                )
            else:
                raise ValueError("Unrecognized data type.")

            # Outlet concentration of each component
            model.ca = pyo.Var(initialize=5000.0, within=pyo.PositiveReals)
            model.cb = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
            model.cc = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
            model.cd = pyo.Var(initialize=1000.0, within=pyo.PositiveReals)

            # Objective
            model.obj = pyo.Objective(expr=model.cb, sense=pyo.maximize)

            # Constraints
            model.ca_bal = pyo.Constraint(
                expr=(
                    0
                    == model.sv * model.caf
                    - model.sv * model.ca
                    - model.k1 * model.ca
                    - 2.0 * model.k3 * model.ca**2.0
                )
            )

            model.cb_bal = pyo.Constraint(
                expr=(
                    0
                    == -model.sv * model.cb + model.k1 * model.ca - model.k2 * model.cb
                )
            )

            model.cc_bal = pyo.Constraint(
                expr=(0 == -model.sv * model.cc + model.k2 * model.cb)
            )

            model.cd_bal = pyo.Constraint(
                expr=(0 == -model.sv * model.cd + model.k3 * model.ca**2.0)
            )

            return model

        # Data from the design
        data = pd.DataFrame(
            data=[
                [1.05, 10000, 3458.4, 1060.8, 1683.9, 1898.5],
                [1.10, 10000, 3535.1, 1064.8, 1613.3, 1893.4],
                [1.15, 10000, 3609.1, 1067.8, 1547.5, 1887.8],
                [1.20, 10000, 3680.7, 1070.0, 1486.1, 1881.6],
                [1.25, 10000, 3750.0, 1071.4, 1428.6, 1875.0],
                [1.30, 10000, 3817.1, 1072.2, 1374.6, 1868.0],
                [1.35, 10000, 3882.2, 1072.4, 1324.0, 1860.7],
                [1.40, 10000, 3945.4, 1072.1, 1276.3, 1853.1],
                [1.45, 10000, 4006.7, 1071.3, 1231.4, 1845.3],
                [1.50, 10000, 4066.4, 1070.1, 1189.0, 1837.3],
                [1.55, 10000, 4124.4, 1068.5, 1148.9, 1829.1],
                [1.60, 10000, 4180.9, 1066.5, 1111.0, 1820.8],
                [1.65, 10000, 4235.9, 1064.3, 1075.0, 1812.4],
                [1.70, 10000, 4289.5, 1061.8, 1040.9, 1803.9],
                [1.75, 10000, 4341.8, 1059.0, 1008.5, 1795.3],
                [1.80, 10000, 4392.8, 1056.0, 977.7, 1786.7],
                [1.85, 10000, 4442.6, 1052.8, 948.4, 1778.1],
                [1.90, 10000, 4491.3, 1049.4, 920.5, 1769.4],
                [1.95, 10000, 4538.8, 1045.8, 893.9, 1760.8],
            ],
            columns=["sv", "caf", "ca", "cb", "cc", "cd"],
        )

        theta_names = ["k1", "k2", "k3"]

        def SSE(model, data):
            expr = (
                (float(data.iloc[0]["ca"]) - model.ca) ** 2
                + (float(data.iloc[0]["cb"]) - model.cb) ** 2
                + (float(data.iloc[0]["cc"]) - model.cc) ** 2
                + (float(data.iloc[0]["cd"]) - model.cd) ** 2
            )
            return expr

        self.pest = parmest.Estimator(reactor_design_model, data, theta_names, SSE)

    def test_scen_from_exps(self):
        scenmaker = sc.ScenarioCreator(self.pest, "ipopt")
        experimentscens = sc.ScenarioSet("Experiments")
        scenmaker.ScenariosFromExperiments(experimentscens)
        experimentscens.write_csv("delme_exp_csv.csv")
        df = pd.read_csv("delme_exp_csv.csv")
        os.remove("delme_exp_csv.csv")
        # March '20: all reactor_design experiments have the same theta values!
        k1val = df.loc[5].at["k1"]
        self.assertAlmostEqual(k1val, 5.0 / 6.0, places=2)
        tval = experimentscens.ScenarioNumber(0).ThetaVals["k1"]
        self.assertAlmostEqual(tval, 5.0 / 6.0, places=2)

    @unittest.skipIf(not uuid_available, "The uuid module is not available")
    def test_no_csv_if_empty(self):
        # low level test of scenario sets
        # verify that nothing is written, but no errors with empty set

        emptyset = sc.ScenarioSet("empty")
        tfile = uuid.uuid4().hex + ".csv"
        emptyset.write_csv(tfile)
        self.assertFalse(
            os.path.exists(tfile), "ScenarioSet wrote csv in spite of empty set"
        )


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestScenarioSemibatchDeprecated(unittest.TestCase):
    def setUp(self):

        import json
        from pyomo.environ import (
            ConcreteModel,
            Set,
            Param,
            Var,
            Constraint,
            ConstraintList,
            Expression,
            Objective,
            TransformationFactory,
            SolverFactory,
            exp,
            minimize,
        )
        from pyomo.dae import ContinuousSet, DerivativeVar

        def generate_model(data):
            # if data is a file name, then load file first
            if isinstance(data, str):
                file_name = data
                try:
                    with open(file_name, "r") as infile:
                        data = json.load(infile)
                except:
                    raise RuntimeError(f"Could not read {file_name} as json")

            # unpack and fix the data
            cameastemp = data["Ca_meas"]
            cbmeastemp = data["Cb_meas"]
            ccmeastemp = data["Cc_meas"]
            trmeastemp = data["Tr_meas"]

            cameas = {}
            cbmeas = {}
            ccmeas = {}
            trmeas = {}
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
            m.R = Param(initialize=8.314)  # kJ/kmol/K
            m.Mwa = Param(initialize=50.0)  # kg/kmol
            m.rhor = Param(initialize=1000.0)  # kg/m^3
            m.cpr = Param(initialize=3.9)  # kJ/kg/K
            m.Tf = Param(initialize=300)  # K
            m.deltaH1 = Param(initialize=-40000.0)  # kJ/kmol
            m.deltaH2 = Param(initialize=-50000.0)  # kJ/kmol
            m.alphaj = Param(initialize=0.8)  # kJ/s/m^2/K
            m.alphac = Param(initialize=0.7)  # kJ/s/m^2/K
            m.Aj = Param(initialize=5.0)  # m^2
            m.Ac = Param(initialize=3.0)  # m^2
            m.Vj = Param(initialize=0.9)  # m^3
            m.Vc = Param(initialize=0.07)  # m^3
            m.rhow = Param(initialize=700.0)  # kg/m^3
            m.cpw = Param(initialize=3.1)  # kJ/kg/K
            m.Ca0 = Param(initialize=data["Ca0"])  # kmol/m^3)
            m.Cb0 = Param(initialize=data["Cb0"])  # kmol/m^3)
            m.Cc0 = Param(initialize=data["Cc0"])  # kmol/m^3)
            m.Tr0 = Param(initialize=300.0)  # K
            m.Vr0 = Param(initialize=1.0)  # m^3

            m.time = ContinuousSet(
                bounds=(0, 21600), initialize=m.measT
            )  # Time in seconds

            #
            # Control Inputs
            #
            def _initTc(m, t):
                if t < 10800:
                    return data["Tc1"]
                else:
                    return data["Tc2"]

            m.Tc = Param(
                m.time, initialize=_initTc, default=_initTc
            )  # bounds= (288,432) Cooling coil temp, control input

            def _initFa(m, t):
                if t < 10800:
                    return data["Fa1"]
                else:
                    return data["Fa2"]

            m.Fa = Param(
                m.time, initialize=_initFa, default=_initFa
            )  # bounds=(0,0.05) Inlet flow rate, control input

            #
            # Parameters being estimated
            #
            m.k1 = Var(initialize=14, bounds=(2, 100))  # 1/s Actual: 15.01
            m.k2 = Var(initialize=90, bounds=(2, 150))  # 1/s Actual: 85.01
            m.E1 = Var(
                initialize=27000.0, bounds=(25000, 40000)
            )  # kJ/kmol Actual: 30000
            m.E2 = Var(
                initialize=45000.0, bounds=(35000, 50000)
            )  # kJ/kmol Actual: 40000
            # m.E1.fix(30000)
            # m.E2.fix(40000)

            #
            # Time dependent variables
            #
            m.Ca = Var(m.time, initialize=m.Ca0, bounds=(0, 25))
            m.Cb = Var(m.time, initialize=m.Cb0, bounds=(0, 25))
            m.Cc = Var(m.time, initialize=m.Cc0, bounds=(0, 25))
            m.Vr = Var(m.time, initialize=m.Vr0)
            m.Tr = Var(m.time, initialize=m.Tr0)
            m.Tj = Var(
                m.time, initialize=310.0, bounds=(288, None)
            )  # Cooling jacket temp, follows coil temp until failure

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

            def _dCacon(m, t):
                if t == 0:
                    return Constraint.Skip
                return (
                    m.dCa[t]
                    == m.Fa[t] / m.Vr[t] - m.k1 * exp(-m.E1 / (m.R * m.Tr[t])) * m.Ca[t]
                )

            m.dCacon = Constraint(m.time, rule=_dCacon)

            def _dCbcon(m, t):
                if t == 0:
                    return Constraint.Skip
                return (
                    m.dCb[t]
                    == m.k1 * exp(-m.E1 / (m.R * m.Tr[t])) * m.Ca[t]
                    - m.k2 * exp(-m.E2 / (m.R * m.Tr[t])) * m.Cb[t]
                )

            m.dCbcon = Constraint(m.time, rule=_dCbcon)

            def _dCccon(m, t):
                if t == 0:
                    return Constraint.Skip
                return m.dCc[t] == m.k2 * exp(-m.E2 / (m.R * m.Tr[t])) * m.Cb[t]

            m.dCccon = Constraint(m.time, rule=_dCccon)

            def _dVrcon(m, t):
                if t == 0:
                    return Constraint.Skip
                return m.dVr[t] == m.Fa[t] * m.Mwa / m.rhor

            m.dVrcon = Constraint(m.time, rule=_dVrcon)

            def _dTrcon(m, t):
                if t == 0:
                    return Constraint.Skip
                return m.rhor * m.cpr * m.dTr[t] == m.Fa[t] * m.Mwa * m.cpr / m.Vr[
                    t
                ] * (m.Tf - m.Tr[t]) - m.k1 * exp(-m.E1 / (m.R * m.Tr[t])) * m.Ca[
                    t
                ] * m.deltaH1 - m.k2 * exp(
                    -m.E2 / (m.R * m.Tr[t])
                ) * m.Cb[
                    t
                ] * m.deltaH2 + m.alphaj * m.Aj / m.Vr0 * (
                    m.Tj[t] - m.Tr[t]
                ) + m.alphac * m.Ac / m.Vr0 * (
                    m.Tc[t] - m.Tr[t]
                )

            m.dTrcon = Constraint(m.time, rule=_dTrcon)

            def _singlecooling(m, t):
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
                return sum(
                    (m.Ca[t] - m.Ca_meas[t]) ** 2
                    + (m.Cb[t] - m.Cb_meas[t]) ** 2
                    + (m.Cc[t] - m.Cc_meas[t]) ** 2
                    + 0.01 * (m.Tr[t] - m.Tr_meas[t]) ** 2
                    for t in m.measT
                )

            def MissingMeasurements(m):
                if data["experiment"] == 1:
                    return sum(
                        (m.Ca[t] - m.Ca_meas[t]) ** 2
                        + (m.Cb[t] - m.Cb_meas[t]) ** 2
                        + (m.Cc[t] - m.Cc_meas[t]) ** 2
                        + (m.Tr[t] - m.Tr_meas[t]) ** 2
                        for t in m.measT
                    )
                elif data["experiment"] == 2:
                    return sum((m.Tr[t] - m.Tr_meas[t]) ** 2 for t in m.measT)
                else:
                    return sum(
                        (m.Cb[t] - m.Cb_meas[t]) ** 2 + (m.Tr[t] - m.Tr_meas[t]) ** 2
                        for t in m.measT
                    )

            m.SecondStageCost = Expression(rule=MissingMeasurements)

            def total_cost_rule(model):
                return model.FirstStageCost + model.SecondStageCost

            m.Total_Cost_Objective = Objective(rule=total_cost_rule, sense=minimize)

            # Discretize model
            disc = TransformationFactory("dae.collocation")
            disc.apply_to(m, nfe=20, ncp=4)
            return m

        # Vars to estimate in parmest
        theta_names = ["k1", "k2", "E1", "E2"]

        self.fbase = os.path.join(testdir, "..", "examples", "semibatch")
        # Data, list of dictionaries
        data = []
        for exp_num in range(10):
            fname = "exp" + str(exp_num + 1) + ".out"
            fullname = os.path.join(self.fbase, fname)
            with open(fullname, "r") as infile:
                d = json.load(infile)
                data.append(d)

        # Note, the model already includes a 'SecondStageCost' expression
        # for the sum of squared error that will be used in parameter estimation

        self.pest = parmest.Estimator(generate_model, data, theta_names)

    def test_semibatch_bootstrap(self):
        scenmaker = sc.ScenarioCreator(self.pest, "ipopt")
        bootscens = sc.ScenarioSet("Bootstrap")
        numtomake = 2
        scenmaker.ScenariosFromBootstrap(bootscens, numtomake, seed=1134)
        tval = bootscens.ScenarioNumber(0).ThetaVals["k1"]
        self.assertAlmostEqual(tval, 20.64, places=1)


if __name__ == "__main__":
    unittest.main()
