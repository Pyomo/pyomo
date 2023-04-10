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
from pyomo.contrib.mpc.examples.cstr.run_mpc import get_steady_state_data, run_cstr_mpc


ipopt_available = pyo.SolverFactory("ipopt").available()


@unittest.skipIf(not ipopt_available, "ipopt is not available")
class TestCSTRMPC(unittest.TestCase):
    # This data was obtained from a run of this code. The test is
    # intended to make sure that values do not change, not that
    # they are correct in some absolute sense.
    _pred_A_data = [
        1.15385,
        2.11629,
        2.59104,
        2.82521,
        2.94072,
        2.99770,
        2.84338,
        2.76022,
        2.71541,
        2.69127,
        2.67826,
        2.70659,
        2.72163,
        2.72961,
        2.73384,
        2.73609,
        2.73100,
        2.72830,
        2.72686,
        2.72609,
        2.72568,
        2.72660,
        2.72709,
        2.72735,
        2.72749,
        2.72756,
    ]
    _pred_B_data = [
        3.85615,
        2.89371,
        2.41896,
        2.18479,
        2.06928,
        2.01230,
        2.16662,
        2.24978,
        2.29459,
        2.31873,
        2.33174,
        2.30341,
        2.28837,
        2.28039,
        2.27616,
        2.27391,
        2.27900,
        2.28170,
        2.28314,
        2.28391,
        2.28432,
        2.28340,
        2.28291,
        2.28265,
        2.28251,
        2.28244,
    ]

    def _get_initial_data(self):
        initial_data = mpc.ScalarData({"flow_in[*]": 0.3})
        return get_steady_state_data(initial_data)

    def _get_setpoint_data(self):
        setpoint_data = mpc.ScalarData({"flow_in[*]": 1.2})
        return get_steady_state_data(setpoint_data)

    def test_mpc_simulation(self):
        initial_data = self._get_initial_data()
        setpoint_data = self._get_setpoint_data()
        sample_time = 2.0
        samples_per_horizon = 5
        ntfe_per_sample = 2
        ntfe_plant = 5
        simulation_steps = 5
        m_plant, sim_data = run_cstr_mpc(
            initial_data,
            setpoint_data,
            samples_per_controller_horizon=samples_per_horizon,
            sample_time=sample_time,
            ntfe_per_sample_controller=ntfe_per_sample,
            ntfe_plant=ntfe_plant,
            simulation_steps=simulation_steps,
        )
        sim_time_points = [
            sample_time / ntfe_plant * i
            for i in range(simulation_steps * ntfe_plant + 1)
        ]

        AB_data = sim_data.extract_variables(
            [m_plant.conc[:, "A"], m_plant.conc[:, "B"]]
        )

        A_cuid = sim_data.get_cuid(m_plant.conc[:, "A"])
        B_cuid = sim_data.get_cuid(m_plant.conc[:, "B"])
        pred_data = {A_cuid: self._pred_A_data, B_cuid: self._pred_B_data}

        self.assertStructuredAlmostEqual(pred_data, AB_data.get_data(), delta=1e-3)
        self.assertStructuredAlmostEqual(
            sim_time_points, AB_data.get_time_points(), delta=1e-7
        )


if __name__ == "__main__":
    unittest.main()
