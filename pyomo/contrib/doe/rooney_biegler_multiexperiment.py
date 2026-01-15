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

"""
Rooney Biegler model, based on Rooney, W. C. and Biegler, L. T. (2001). Design for
model parameter uncertainty using nonlinear confidence regions. AIChE Journal,
47(8), 1794-1804.
"""

import pyomo.environ as pyo
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.common.dependencies import pandas as pd, numpy as np, matplotlib


def rooney_biegler_model(data, theta=None):
    model = pyo.ConcreteModel()

    if theta is None:
        theta = {'asymptote': 15, 'rate_constant': 0.5}

    model.asymptote = pyo.Var(initialize=theta['asymptote'])
    model.rate_constant = pyo.Var(initialize=theta['rate_constant'])

    # Fix the unknown parameters
    model.asymptote.fix()
    model.rate_constant.fix()

    # Add the experiment inputs
    model.hour = pyo.Var(initialize=data["hour"].iloc[0], bounds=(0, 10))

    # Fix the experiment inputs
    model.hour.fix()

    # Add the response variable
    model.y = pyo.Var(within=pyo.PositiveReals, initialize=data["y"].iloc[0])

    def response_rule(m):
        return m.y == m.asymptote * (1 - pyo.exp(-m.rate_constant * m.hour))

    model.response_function = pyo.Constraint(rule=response_rule)

    return model


class RooneyBieglerExperiment(Experiment):

    def __init__(self, data, measure_error=None, theta=None):
        self.data = data
        self.model = None
        self.measure_error = measure_error
        self.theta = theta

    def create_model(self):
        # rooney_biegler_model expects a dataframe
        data_df = self.data.to_frame().transpose()
        self.model = rooney_biegler_model(data_df, theta=self.theta)

    def label_model(self):

        m = self.model

        # Add experiment outputs as a suffix
        # Experiment outputs suffix is required for parmest
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, self.data['y'])])

        # Add unknown parameters as a suffix
        # Unknown parameters suffix is required for both Pyomo.DoE and parmest
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.value(k)) for k in [m.asymptote, m.rate_constant]
        )

        # Add measurement error as a suffix
        # Measurement error suffix is required for Pyomo.DoE and
        #  `cov` estimation in parmest
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, self.measure_error)])

        # Add hour as an experiment input
        # Experiment inputs suffix is required for Pyomo.DoE
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update([(m.hour, self.data['hour'])])

        # For multiple experiments, we need to add symmetry breaking constraints
        # to avoid identical models as a suffix `sym_break_cons`
        m.sym_break_cons = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.sym_break_cons[m.hour] = None

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


def run_rooney_biegler_doe(
    optimize_experiment_A=False,
    optimize_experiment_D=True,
    compute_FIM_full_factorial=False,
    draw_factorial_figure=False,
    design_range={'hour': [0, 10, 40]},
    plot_optimal_design=False,
    tee=False,
    print_output=False,
):
    # Initialize a container for all potential results
    results_container = {
        "experiment": None,
        "results_dict": {},
        "optimization": {},  # Will hold D/A optimization results if run
        "plots": [],  # Store figure objects if created
    }

    # Data Setup
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )
    theta = {'asymptote': 15, 'rate_constant': 0.5}
    measurement_error = 0.1

    # Compute prior FIM from existing data - COMMENTED OUT FOR TESTING
    # Use a hardcoded prior FIM from previous successful run
    FIM = np.array([[368.87, 3410.37], [3410.37, 41928.45]])

    # Base Experiment for Design
    rooney_biegler_experiment = RooneyBieglerExperiment(
        data=data.loc[0, :], theta=theta, measure_error=measurement_error
    )
    results_container["experiment"] = rooney_biegler_experiment

    rooney_biegler_doe = DesignOfExperiments(
        experiment_list=[rooney_biegler_experiment],
        objective_option="determinant",
        tee=tee,
        prior_FIM=FIM,
        # improve_cholesky_roundoff_error=True,
    )

    if optimize_experiment_D:
        # D-Optimality
        # Create a custom solver without MA57 to avoid crashes
        solver = pyo.SolverFactory("ipopt")
        solver.options["linear_solver"] = "mumps"  # Use mumps instead of ma57
        solver.options["halt_on_ampl_error"] = "yes"
        solver.options["max_iter"] = 3000

        rooney_biegler_doe_D = DesignOfExperiments(
            experiment_list=[rooney_biegler_experiment],
            objective_option="determinant",
            tee=True,  # Enable verbose output to see solver details
            prior_FIM=FIM,
            solver=solver,  # Use custom solver
            # improve_cholesky_roundoff_error=True,
        )
        # COMMENTED OUT FOR TESTING - Skip single experiment optimization
        # rooney_biegler_doe_D.run_doe()

        # results_container["optimization"]["D"] = {
        #     "value": rooney_biegler_doe_D.results['log10 D-opt'],
        #     "design": rooney_biegler_doe_D.results['Experiment Design'],
        # }
        # if print_output:
        #     print("Optimal results for D-optimality:", rooney_biegler_doe_D.results)

        print("\n" * 2, "Multiple experiment code started")
        # Test multi-experiment optimization
        rooney_biegler_doe_D.optimize_experiments(n_exp=1)

        print("\nMulti-experiment optimization completed!")
        print("Results:", rooney_biegler_doe_D.results)

    # COMMENTED OUT FOR TESTING
    # if optimize_experiment_A:
    #     # A-Optimality
    #     rooney_biegler_doe_A = DesignOfExperiments(
    #         experiment_list=[rooney_biegler_experiment],
    #         objective_option="trace",
    #         tee=tee,
    #         prior_FIM=FIM,
    #         # improve_cholesky_roundoff_error=False,
    #     )
    #     rooney_biegler_doe_A.run_doe()

    #     results_container["optimization"]["A"] = {
    #         "value": rooney_biegler_doe_A.results['log10 A-opt'],
    #         "design": rooney_biegler_doe_A.results['Experiment Design'],
    #     }

    #     if print_output:
    #         print("Optimal results for A-optimality:", rooney_biegler_doe_A.results)

    # COMMENTED OUT FOR TESTING
    # # Compute Full Factorial Design Results
    # if compute_FIM_full_factorial:
    #     results_container["results_dict"] = (
    #         rooney_biegler_doe.compute_FIM_full_factorial(design_ranges=design_range)
    #     )
    #     if print_output:
    #         print("Full Factorial Design Results:\n", results_container["results_dict"])

    # COMMENTED OUT FOR TESTING
    # # Custom Plotting Functionality that shows the optimal designs and
    # # the max/min from the full factorial results
    # # Plotting Block
    if False and plot_optimal_design:
        plt = matplotlib.pyplot
        res_dict = results_container["results_dict"]
        opt_D = results_container["optimization"]["D"]
        opt_A = results_container["optimization"]["A"]

        # D-Optimality Plot
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        # Locate Star values (max)
        id_max_D = np.argmax(res_dict["log10 D-opt"])

        ax1.plot(res_dict["hour"], res_dict["log10 D-opt"])
        ax1.scatter(
            opt_D["design"][0],
            opt_D["value"],
            color='green',
            marker="*",
            s=600,
            label=f'Optimal D-opt: {opt_D["value"]:.2f}',
        )
        ax1.scatter(
            res_dict["hour"][id_max_D],
            res_dict["log10 D-opt"][id_max_D],
            color='red',
            marker="o",
            s=200,
            label='Max D-opt in Range',
        )
        ax1.set_xlabel("Time Points")
        ax1.set_ylabel("log10 D-optimality")
        ax1.legend()

        results_container["plots"].append(fig1)

        # A-Optimality Plot
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        # Locate Star values (min)
        id_min_A = np.argmin(res_dict["log10 A-opt"])

        ax2.plot(res_dict["hour"], res_dict["log10 A-opt"])
        ax2.scatter(
            opt_A["design"][0],
            opt_A["value"],
            color='green',
            marker="*",
            s=600,
            label=f'Optimal A-opt: {opt_A["value"]:.2f}',
        )
        ax2.scatter(
            res_dict["hour"][id_min_A],
            res_dict["log10 A-opt"][id_min_A],
            color='red',
            marker="o",
            s=200,
            label='Min A-opt in Range',
        )
        ax2.set_xlabel("Time Points")
        ax2.set_ylabel("log10 A-optimality")
        ax2.legend()

        results_container["plots"].append(fig2)

    # COMMENTED OUT FOR TESTING
    # if draw_factorial_figure:
    #     rooney_biegler_doe.draw_factorial_figure(
    #         sensitivity_design_variables=['hour'],
    #         fixed_design_variables={},
    #         log_scale=False,
    #         figure_file_name="rooney_biegler",
    #     )

    return results_container


if __name__ == "__main__":

    results = run_rooney_biegler_doe(
        optimize_experiment_A=False,  # Skip for testing
        optimize_experiment_D=True,  # Focus on this
        compute_FIM_full_factorial=False,  # Skip for testing
        draw_factorial_figure=False,  # Skip for testing
        design_range={'hour': [0, 10, 11]},
        plot_optimal_design=False,  # Skip for testing
        print_output=True,
    )

    # Show plots if running locally
    # matplotlib.pyplot.show()
