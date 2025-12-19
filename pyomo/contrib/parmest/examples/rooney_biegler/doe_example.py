from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
    RooneyBieglerExperiment,
)
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.common.dependencies import pandas as pd, numpy as np, matplotlib


def run_rooney_biegler_doe(
    optimize_experiment_A=False,
    optimize_experiment_D=False,
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

    # Compute prior FIM from existing data
    FIM = np.zeros((2, 2))
    for i in range(len(data)):
        exp_i = RooneyBieglerExperiment(
            data=data.loc[i, :], theta=theta, measure_error=measurement_error
        )
        doe_prior = DesignOfExperiments(
            experiment=exp_i, objective_option="determinant", tee=tee
        )
        FIM += doe_prior.compute_FIM()

    # Base Experiment for Design
    rooney_biegler_experiment = RooneyBieglerExperiment(
        data=data.loc[0, :], theta=theta, measure_error=measurement_error
    )
    results_container["experiment"] = rooney_biegler_experiment

    rooney_biegler_doe = DesignOfExperiments(
        experiment=rooney_biegler_experiment,
        objective_option="determinant",
        tee=tee,
        prior_FIM=FIM,
    )

    if optimize_experiment_D:
        # D-Optimality
        rooney_biegler_doe_D = DesignOfExperiments(
            experiment=rooney_biegler_experiment,
            objective_option="determinant",
            tee=tee,
            prior_FIM=FIM,
        )
        rooney_biegler_doe_D.run_doe()

        results_container["optimization"]["D"] = {
            "value": rooney_biegler_doe_D.results['log10 D-opt'],
            "design": rooney_biegler_doe_D.results['Experiment Design'],
        }
        if print_output:
            print("Optimal results for D-optimality:", rooney_biegler_doe_D.results)

    if optimize_experiment_A:
        # A-Optimality
        rooney_biegler_doe_A = DesignOfExperiments(
            experiment=rooney_biegler_experiment,
            objective_option="trace",
            tee=tee,
            prior_FIM=FIM,
        )
        rooney_biegler_doe_A.run_doe()

        results_container["optimization"]["A"] = rooney_biegler_doe_A.results
        if print_output:
            print("Optimal results for A-optimality:", rooney_biegler_doe_A.results)

    # Compute Full Factorial Design Results
    if compute_FIM_full_factorial:
        results_container["results_dict"] = (
            rooney_biegler_doe.compute_FIM_full_factorial(design_ranges=design_range)
        )
        if print_output:
            print("Full Factorial Design Results:\n", results_container["results_dict"])

    # Plotting Block
    if plot_optimal_design:
        plt = matplotlib.pyplot
        res_dict = results_container["results_dict"]
        opt_D = results_container["optimization"]["D"]
        opt_A = results_container["optimization"]["A"]

        # D-Optimality Plot
        fig1, ax1 = plt.subplots(figsize=(10, 5))

        # Locate Star values (max/min)
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

    if draw_factorial_figure:
        rooney_biegler_doe.draw_factorial_figure(
            sensitivity_design_variables=['hour'],
            fixed_design_variables={},
            log_scale=False,
            figure_file_name="rooney_biegler",
        )

    return results_container
