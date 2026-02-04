#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


# funnel.py – scalar funnel helper (no Filter list)
# --------------------------------------------------------
# Implements Hameed et al. (https://arxiv.org/abs/2511.18998) funnel logic
# Funnel is an alternative to filter globalization mechanism
# This addition lets the users to choose Funnel or Filter
# Public API (mirrors simplicity of filterMethod):
#   funnel = Funnel(phi_init, f_best_init,
#                   phi_min, kappa_f, alpha, beta, mu_s, eta)
#
#   status = funnel.classify_step(theta_k, theta_plus,
#                                 f_k, f_plus, trust_radius)
#           # returns 'f', 'theta', or 'reject'
#
#   if status == 'f':
#       funnel.accept_f(theta_plus, f_plus)
#   elif status == 'theta':
#       funnel.accept_theta(theta_plus)
#
# The class stores only two scalars (phi, f_best) and imposes the
# three Kiessling tests:
#   • switching   f_k - f⁺ ≥ μ_s (θ_k - θ⁺)
#   • Armijo      f_k - f⁺ ≥ η₁ Δ⁽ᵏ⁾          (Δ supplied by caller)
#   • θ‑shrink    θ⁺ ≤ β φᵏ
# --------------------------------------------------------

from __future__ import annotations


class Funnel:
    """Scalar funnel tracker for Trust‑Region Funnel (grey‑box).

    Parameters
    ----------
    phi_init     : initial funnel width  φ⁰  (≥ θ⁰)
    f_best_init  : first feasible objective (usually f⁰)
    phi_min      : hard floor on φ           (>0)
    kappa_f      : shrink factor after f‑step (0<κ_f<1)
    kappa_r      : relax factor for theta (>1)
    alpha        : curvature exponent        (0<α<1)
    beta         : θ‑type shrink factor      (0<β<1)
    mu_s         : switching parameter δ     (small, e.g.1e‑2)
    eta          : Armijo parameter          (0<η<1)
    """

    # -----------------------------------------------------
    def __init__(
        self,
        phi_init: float,
        f_best_init: float,
        phi_min: float,
        kappa_f: float,
        kappa_r: float,
        alpha: float,
        beta: float,
        mu_s: float,
        eta: float,
    ):
        self.phi = max(phi_min, phi_init)
        self.f_best = f_best_init
        # store parameters
        self.phi_min = phi_min
        self.kappa_f = kappa_f
        self.kappa_r = kappa_r
        self.alpha = alpha
        self.beta = beta
        self.mu_s = mu_s
        self.eta = eta

    # -----------------------------------------------------
    #   Helper tests (all scalar, no surrogates required)
    # -----------------------------------------------------
    def _inside_funnel(self, theta_new: float) -> bool:
        return theta_new <= self.phi

    def _switching(
        self, f_old: float, f_new: float, theta_old: float, theta_new: float
    ) -> bool:
        # Δf ≥ μ_s · Δθ
        return (f_old - f_new) >= self.mu_s * ((theta_old) ** 2)
        # return (f_old - f_new) >= self.mu_s * (theta_old - theta_new)

    def _armijo(self, f_old: float, f_new: float, delta: float) -> bool:
        # actual reduction ≥ η₁ Δ (trust‑region radius used as scale)
        return (f_old - f_new) >= self.eta * delta

    def _theta_shrink(self, theta_new: float) -> bool:
        return theta_new <= self.beta * self.phi

    # -----------------------------------------------------
    #   Public classifier
    # -----------------------------------------------------
    def classify_step(
        self,
        theta_old: float,
        theta_new: float,
        f_old: float,
        f_new: float,
        delta: float,
    ) -> str:
        """Return 'f', 'theta', or 'reject' for the trial point."""
        # theta, f and reject steps
        if self._inside_funnel(theta_new):
            # candidate f‑step → need Armijo
            if self._switching(f_old, f_new, theta_old, theta_new):
                return 'f' if self._armijo(f_old, f_new, delta) else 'reject'
            # else candidate θ‑step → need θ‑shrink
            return 'theta' if self._theta_shrink(theta_new) else 'reject'

        # Outside funnel: allow relaxed theta step
        if (
            self._switching(f_old, f_new, theta_old, theta_new)
            and theta_new <= self.kappa_r * self.phi
        ):
            return 'theta-relax'

        return 'reject'

    # -----------------------------------------------------
    #   Updates after acceptance
    # -----------------------------------------------------
    def accept_f(self, theta_new: float, f_new: float):
        """Call after accepting an f‑type step."""
        if f_new < self.f_best:
            self.f_best = f_new

    def accept_theta(self, theta_new: float):
        """Call after accepting a θ‑type step."""
        kf = self.kappa_f
        # gentle convex combo shrink
        self.phi = max(self.phi_min, (1 - kf) * theta_new + kf * self.phi)
