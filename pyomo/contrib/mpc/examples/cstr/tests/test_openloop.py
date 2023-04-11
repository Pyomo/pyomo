#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.run_openloop import run_cstr_openloop


ipopt_available = pyo.SolverFactory("ipopt").available()


@unittest.skipIf(not ipopt_available, "ipopt is not available")
class TestCSTROpenLoop(unittest.TestCase):
    # This data was obtained from a run of this code. The test is
    # intended to make sure that values do not change, not that
    # they are correct in some absolute sense.
    _pred_A_data = [
        1.00000,
        1.50000,
        1.83333,
        2.05556,
        2.20370,
        2.30247,
        2.36831,
        2.41221,
        2.44147,
        2.46098,
        2.47399,
        2.48266,
        2.48844,
        2.36031,
        2.27039,
        2.20729,
        2.16301,
        2.13194,
        2.11013,
        2.09483,
        2.08409,
        2.07656,
        2.07127,
        2.06756,
        2.06495,
        2.16268,
        2.22893,
        2.27385,
        2.30431,
        2.32495,
        2.33895,
        2.34844,
        2.35488,
        2.35924,
        2.36220,
        2.36420,
        2.36556,
        2.36648,
        2.36711,
        2.36753,
        2.36782,
    ]
    _pred_B_data = [
        0.00000,
        0.30200,
        0.61027,
        0.90132,
        1.16380,
        1.39353,
        1.59049,
        1.75683,
        1.89576,
        2.01081,
        2.10544,
        2.18289,
        2.24600,
        2.41517,
        2.54001,
        2.63284,
        2.70242,
        2.75502,
        2.79516,
        2.82605,
        2.85007,
        2.86890,
        2.88380,
        2.89569,
        2.90526,
        2.81484,
        2.75455,
        2.71450,
        2.68802,
        2.67062,
        2.65927,
        2.65195,
        2.64728,
        2.64436,
        2.64258,
        2.64153,
        2.64096,
        2.64067,
        2.64057,
        2.64058,
        2.64064,
    ]

    def _get_input_sequence(self):
        input_sequence = mpc.TimeSeriesData(
            {"flow_in[*]": [0.1, 1.0, 0.7, 0.9]}, [0.0, 3.0, 6.0, 10.0]
        )
        return mpc.data.convert.series_to_interval(input_sequence)

    def test_openloop_simulation(self):
        input_sequence = self._get_input_sequence()
        ntfe = 4
        model_horizon = 1.0
        simulation_steps = 10
        m, sim_data = run_cstr_openloop(
            input_sequence, model_horizon=1.0, ntfe=4, simulation_steps=10
        )
        sim_time_points = [
            model_horizon / ntfe * i for i in range(simulation_steps * ntfe + 1)
        ]

        AB_data = sim_data.extract_variables([m.conc[:, "A"], m.conc[:, "B"]])

        A_cuid = sim_data.get_cuid(m.conc[:, "A"])
        B_cuid = sim_data.get_cuid(m.conc[:, "B"])
        pred_data = {A_cuid: self._pred_A_data, B_cuid: self._pred_B_data}

        self.assertStructuredAlmostEqual(pred_data, AB_data.get_data(), delta=1e-3)
        self.assertStructuredAlmostEqual(
            sim_time_points, AB_data.get_time_points(), delta=1e-7
        )


if __name__ == "__main__":
    unittest.main()
