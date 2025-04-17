import numpy as np
# from pyomo.common.dependencies import numpy as np
import pytest

def compute_FIM_metrics(FIM):
    # Compute and record metrics on FIM
    det_FIM = np.linalg.det(FIM)   # Determinant of FIM         
    D_opt = np.log10(det_FIM)
    trace_FIM = np.trace(FIM)  # Trace of FIM
    A_opt = np.log10(trace_FIM)  
    E_vals, E_vecs =np.linalg.eig(FIM)  # Grab eigenvalues and eigenvectors

    E_ind = np.argmin(E_vals.real)  # Grab index of minima to check imaginary
    IMG_THERESHOLD = 1e-6  # Instead of creating a new constant, use `SMALL_DIFF` by importiing it form `doe.py`
    # Warn the user if there is a ``large`` imaginary component (should not be)
    if abs(E_vals.imag[E_ind]) > IMG_THERESHOLD:
        print(
            f"Eigenvalue has imaginary component greater than {IMG_THERESHOLD}, contact developers if this issue persists."
        )

    # If the real value is less than or equal to zero, set the E_opt value to nan
    if E_vals.real[E_ind] <= 0:
        E_opt = np.nan  
    else:
        E_opt = np.log10(E_vals.real[E_ind])

    ME_opt = np.log10(np.linalg.cond(FIM))

    return {
        "det_FIM": det_FIM,
        "trace_FIM": trace_FIM,
        "E_vals": E_vals,
        "D_opt": D_opt,
        "A_opt": A_opt,
        "E_opt": E_opt,
        "ME_opt": ME_opt
    }

def test_FIM_metrics():
    # Create a sample Fisher Information Matrix (FIM)
    FIM = np.array([[4, 2], [2, 3]])

    # Call the function to compute metrics
    results = compute_FIM_metrics(FIM)

    # Use known values for assertions
    det_expected = np.linalg.det(FIM)
    D_opt_expected = np.log10(det_expected)

    trace_expected = np.trace(FIM)
    A_opt_expected = np.log10(trace_expected)

    E_vals_expected, _ = np.linalg.eig(FIM)
    min_eigval = np.min(E_vals_expected.real)

    cond_expected = np.linalg.cond(FIM)

    assert np.isclose(results['det_FIM'], det_expected)
    assert np.isclose(results['trace_FIM'], trace_expected)
    assert np.allclose(results['E_vals'], E_vals_expected)
    assert np.isclose(results['D_opt'], D_opt_expected)
    assert np.isclose(results['A_opt'], A_opt_expected)
    if min_eigval.real > 0:
        assert np.isclose(results['E_opt'], np.log10(min_eigval))
    else:
        assert np.isnan(results['E_opt'])

    assert np.isclose(results['ME_opt'], np.log10(cond_expected))

   

def test_FIM_metrics_warning_printed(capfd):
    # Create a matrix with an imaginary component large enough to trigger the warning
    FIM = np.array([
        [9, -2],
        [9, 3]
        ])

    # Call the function
    compute_FIM_metrics(FIM)

    # Capture stdout and stderr
    out, err = capfd.readouterr()

    # Correct expected message
    expected_message = "Eigenvalue has imaginary component greater than 1e-06, contact developers if this issue persists."

    # Ensure expected message is in the output
    assert expected_message in out
