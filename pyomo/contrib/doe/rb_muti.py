# NO NEED TO CHECK THIS SCRIPT. THIS IS FOR ME TO UNDERSTAND THE RESULT AND WILL BE DELETED LATER.
"""Utility for scanning Rooney-Biegler multi-experiment FIM metrics."""

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    matplotlib,
    matplotlib_available,
)

from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.doe.tests.experiment_class_example_flags import (
    RooneyBieglerMultiExperiment,
)


def rb_multi(hour: np.ndarray, n_exp: int, prior_FIM):
    """
    Compute Rooney-Biegler 2-experiment FIM metrics over hour pairs.

    Parameters
    ----------
    hour:
        Candidate experiment times. A 1-D array is expected; other shapes are
        flattened in row-major order.
    n_exp:
        Number of experiments selected in each combination. This utility is
        intentionally restricted to ``n_exp == 2`` for now.
    prior_FIM:
        Prior information matrix added once to every selected combination.

    Returns
    -------
    dict
        Dictionary containing the hour grid, pairwise total FIMs, the four
        requested log10 metric matrices, and the matplotlib figure.
    """
    if not numpy_available:
        raise ImportError("rb_muti requires numpy.")

    hour = np.asarray(hour, dtype=float).ravel()
    if hour.size == 0:
        raise ValueError("`hour` must contain at least one candidate point.")
    if n_exp != 2:
        raise ValueError(
            f"`rb_muti` currently supports only `n_exp == 2`, got {n_exp!r}."
        )
    if n_exp > hour.size:
        raise ValueError(
            f"`n_exp`={n_exp} cannot exceed the number of candidate hours "
            f"({hour.size})."
        )

    # Compute one single-experiment FIM per candidate hour and reuse it across
    # every combination. This avoids repeating the expensive DOE solve for the
    # same hour value.
    point_fims = []
    for hour_value in hour:
        experiment = RooneyBieglerMultiExperiment(hour=float(hour_value))
        doe = DesignOfExperiments(
            experiment=experiment, objective_option="zero", step=1e-2, prior_FIM=None
        )
        point_fims.append(np.asarray(doe.compute_FIM(method="sequential"), dtype=float))

    fim_shape = point_fims[0].shape
    prior_FIM = np.asarray(prior_FIM, dtype=float)
    if prior_FIM.shape != fim_shape:
        raise ValueError(
            f"`prior_FIM` must have shape {fim_shape}, got {prior_FIM.shape}."
        )

    n_hours = hour.size
    total_fims = np.empty((n_hours, n_hours), dtype=object)
    log10_det = np.empty((n_hours, n_hours), dtype=float)
    log10_trace_inv = np.empty((n_hours, n_hours), dtype=float)
    log10_min_eig = np.empty((n_hours, n_hours), dtype=float)
    log10_cond = np.empty((n_hours, n_hours), dtype=float)

    # Build the full hour-by-hour grid so the diagonal corresponds to running
    # both experiments at the same candidate hour.
    for i in range(n_hours):
        for j in range(n_hours):
            # if j < i:
            #     continue  # Skip lower triangle; FIMs and metrics are symmetric in (i, j)
            total_fim = prior_FIM.copy() + point_fims[i] + point_fims[j]
            total_fims[i, j] = total_fim

            sign, logdet = np.linalg.slogdet(total_fim)
            log10_det[i, j] = logdet / np.log(10.0) if sign > 0 else np.nan

            eigvals = np.linalg.eigvalsh(total_fim)
            min_eig = np.min(eigvals)
            log10_min_eig[i, j] = np.log10(min_eig) if min_eig > 0 else np.nan

            try:
                trace_inv = np.trace(np.linalg.inv(total_fim))
                log10_trace_inv[i, j] = (
                    np.log10(trace_inv)
                    if np.isfinite(trace_inv) and trace_inv > 0
                    else np.nan
                )
            except np.linalg.LinAlgError:
                log10_trace_inv[i, j] = np.nan

            cond_value = np.linalg.cond(total_fim)
            log10_cond[i, j] = (
                np.log10(cond_value)
                if np.isfinite(cond_value) and cond_value > 0
                else np.nan
            )

    if not matplotlib_available:
        raise ImportError("Plotting rb_muti results requires matplotlib.")

    figure, axes = matplotlib.pyplot.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    metric_specs = [
        ("log10(det(total_FIM))", log10_det, np.nanargmax),
        ("log10(trace(inv(total_FIM)))", log10_trace_inv, np.nanargmin),
        ("log10(min eig(total_FIM))", log10_min_eig, np.nanargmax),
        ("log10(cond(total_FIM))", log10_cond, np.nanargmin),
    ]

    for ax, (title, values, selector) in zip(axes, metric_specs):
        image = ax.imshow(values, origin="lower", aspect="auto")
        ax.set_title(title)
        ax.set_ylabel("hour_1")
        ax.set_xlabel("hour_2")
        figure.colorbar(image, ax=ax)

        if np.isfinite(values).any():
            best_index = int(selector(values))
            best_row, best_col = np.unravel_index(best_index, values.shape)
            best_hour_1 = float(hour[best_row])
            best_hour_2 = float(hour[best_col])
            best_value = float(values[best_row, best_col])

            ax.plot(best_col, best_row, marker="*", color="red", markersize=14)
            # Keep the annotation box inside the axes so corner optima do not
            # push the label outside the subplot or into neighboring panels.
            ax.text(
                0.03,
                0.97,
                f"hours=({best_hour_1:.5g}, {best_hour_2:.5g})\nvalue={best_value:.5f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="red",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8},
            )

    for ax in axes:
        ax.set_xticks(np.arange(n_hours))
        ax.set_xticklabels([f"{val:.3g}" for val in hour], rotation=45, ha="right")
        ax.set_yticks(np.arange(n_hours))
        ax.set_yticklabels([f"{val:.3g}" for val in hour])

    figure.tight_layout()
    matplotlib.pyplot.show()

    return {
        "hours": hour,
        "total_fims": total_fims,
        "log10_det": log10_det,
        "log10_trace_inv": log10_trace_inv,
        "log10_min_eig": log10_min_eig,
        "log10_cond": log10_cond,
        "figure": figure,
    }


if __name__ == "__main__":
    candidate_hours = np.linspace(1, 10, 50)
    candidate_hours = np.concatenate([candidate_hours, [1.9321985035514362]])
    prior_information_matrix = np.array(
        [[15.48181217, 357.97684273], [357.97684273, 8277.28811613]]
    )
    prior_information_matrix = np.eye(2)
    exp_list = [
        RooneyBieglerMultiExperiment(hour=2.1, y=8.3),
        RooneyBieglerMultiExperiment(hour=10, y=10.3),
    ]
    from pyomo.opt import SolverFactory

    grey_box_solver = SolverFactory("cyipopt")
    grey_box_solver.config.options["linear_solver"] = "ma57"
    grey_box_solver.config.options['tol'] = 1e-6
    grey_box_solver.config.options['mu_strategy'] = "monotone"

    doe = DesignOfExperiments(
        experiment=exp_list,
        objective_option="trace",
        step=1e-2,
        use_grey_box_objective=True,
        grey_box_solver=grey_box_solver,
        grey_box_tee=False,
    )
    doe.optimize_experiments()
    print("Optimal experiment design:")
    print(doe.results)

    scenario = doe.results["Scenarios"][0]
    got_hours = sorted(exp["Experiment Design"][0] for exp in scenario["Experiments"])
    expected_hours = [1.9321985035514362, 9.999999685577139]
    results = rb_multi(candidate_hours, n_exp=2, prior_FIM=prior_information_matrix)
