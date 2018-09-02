import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class UnconstrainedLineSearch(object):

    def __init__(self, objective_rule, gradient_rule):
        self._opt_rule = objective_rule
        self._grad_rule = gradient_rule
        self._n_backtrack = 0
        logger.info("create UnconstrainedLineSearch object")

    @property
    def num_backtrack(self):
        return self._n_backtrack

    def f(self, x):
        return self._opt_rule(x)

    def gf(self, x):
        return self._grad_rule(x)

    def search(self, x, dx, max_alpha=1.0, rho=0.5, eta1=1e-4, max_iter=40, **kwargs):

        eta2 = kwargs.pop('eta2', 0.9)
        second_condition = kwargs.pop('cond', 0)
        alpha = max_alpha
        self._n_backtrack = 0
        opt = self.f(x)
        grad = self.gf(x)
        prod = grad.dot(dx)

        # check for very small search directions
        if np.linalg.norm(np.absolute(dx) / (1 + np.absolute(x)), ord=np.inf) < 10 * np.finfo(float).eps:
            return alpha

        # only armijo
        if second_condition == 0:
            for i in range(max_iter):
                x_trial = x + alpha * dx
                opt_p = self.f(x_trial)
                if opt_p <= opt + eta1*alpha*grad.dot(dx):
                    return alpha
                alpha *= rho
                self._n_backtrack += 1
        # armijo and wolfe
        elif second_condition == 1:
            for i in range(max_iter):
                x_trial = x + alpha * dx
                opt_p = self.f(x_trial)
                grad_p = self.gf(x_trial)
                if opt_p <= opt + eta1 * alpha * prod and \
                        grad_p.dot(dx) >= eta2 * prod:
                    return alpha
                alpha *= rho
                self._n_backtrack += 1
        # armijo and strong wolfe
        elif second_condition == 2:
            for i in range(max_iter):
                x_trial = x + alpha * dx
                opt_p = self.f(x_trial)
                grad_p = self.gf(x_trial)
                if opt_p <= opt + eta1 * alpha * prod and \
                                abs(grad_p.dot(dx)) >= eta2 * abs(prod):
                    return alpha
                alpha *= rho
                self._n_backtrack += 1
        # armijo and goldstein
        elif second_condition == 3:
            for i in range(max_iter):
                x_trial = x + alpha * dx
                opt_p = self.f(x_trial)
                if opt + (1-eta1)*alpha*prod <= opt_p <= opt + eta1 * alpha * prod :
                    return alpha
                alpha *= rho
                self._n_backtrack += 1
        else:
            raise NotImplementedError('line search conditions not implemented')
        raise RuntimeError('Line search failed')


class BasicFilterLineSearch(object):

    def __init__(self, optimality_rule, feasibility_rule, gradient_rule):

        self._opt_rule = optimality_rule
        self._feas_rule = feasibility_rule
        self._grad_rule = gradient_rule
        self._filter = list()
        self._n_backtrack = 0
        self._theta_min = None
        self._theta_max = None

    @property
    def size_filter(self):
        return len(self._filter)

    @property
    def num_backtrack(self):
        return self._n_backtrack

    def evaluate_optimality(self, x):
        return self._opt_rule(x)

    def evaluate_fesibility(self, x):
        return self._feas_rule(x)

    def evaluate_grad_optimality(self, x):
        return self._grad_rule(x)

    def add_to_filter(self, optimality, feasibility):
        if self.size_filter == 0:
            self._theta_min = 1e-4 * max(1, feasibility)
            self._theta_max = 1e4 * max(1, feasibility)
        self._filter.append((optimality, feasibility))

    def search(self, x, dx, **kwargs):

        max_iter = kwargs.pop('max_backtrack', 40)
        rho = kwargs.pop('rho', 0.5)
        max_alpha = kwargs.pop('max_alpha', 1.0)
        s_phi = kwargs.pop('s_phi', 2.3)
        s_theta = kwargs.pop('s_theta', 1.1)
        kronecker = kwargs.pop('kronecker', 1.0)
        eta_phi = kwargs.pop('eta_phi', 1e-4)
        gamma_theta = kwargs.pop('gamma_theta', 1e-5)
        gamma_phi = kwargs.pop('gamma_phi', 1e-5)

        if self.size_filter == 0:
            raise RuntimeError('Filter must be initialized with at least one pair before calling search')


        alpha = max_alpha
        self._n_backtrack = 0

        # check for very small search directions
        if np.linalg.norm(np.absolute(dx)/(1+np.absolute(x)), ord=np.inf) < 10 * np.finfo(float).eps:
            return alpha

        opt = self.evaluate_optimality(x)
        feas = self.evaluate_fesibility(x)
        grad_opt = self.evaluate_grad_optimality(x)
        mk = grad_opt.dot(dx)

        for i in range(max_iter):
            x_trial = x + alpha * dx
            opt_p = self.evaluate_optimality(x_trial)
            feas_p = self.evaluate_fesibility(x_trial)

            # check if is larger than max infeasibility
            if feas_p >= self._theta_max:
                self._n_backtrack += 1
                alpha *= rho
                continue

            # check if trial is in the filter
            for pair in self._filter:
                if opt_p >= pair[0] and feas_p >= pair[1]:
                    self._n_backtrack += 1
                    alpha *= rho
                    continue

            # switching condition
            if feas <= self._theta_min:
                if mk < 0.0:
                    lhs = alpha * (-mk) ** s_phi
                    rhs = kronecker * feas ** s_theta
                    if lhs > rhs:
                        if opt_p <= opt + eta_phi * alpha * mk:
                            return alpha
                    else:
                        # check SDC or SDO
                        if feas_p <= (1 - gamma_theta) * feas or opt_p <= opt - gamma_phi * feas:
                            self.add_to_filter(opt_p, feas_p)
                            return alpha
                else:
                    # check SDC or SDO
                    if feas_p <= (1 - gamma_theta) * feas or opt_p <= opt - gamma_phi * feas:
                        self.add_to_filter(opt_p, feas_p)
                        return alpha
            else:
                # check SDC or SDO
                if feas_p <= (1 - gamma_theta) * feas or opt_p <= opt - gamma_phi * feas:
                    self.add_to_filter(opt_p, feas_p)
                    return alpha
            self._n_backtrack += 1
            alpha *= rho
        raise RuntimeError('Line search failed. Restoration and soc not implemented yet')

    def search2(self, x, dx, **kwargs):

        max_iter = kwargs.pop('max_backtrack', 10)
        rho = kwargs.pop('rho', 0.5)
        max_alpha = kwargs.pop('max_alpha', 1.0)
        s_phi = kwargs.pop('s_phi', 2.3)
        s_theta = kwargs.pop('s_theta', 1.1)
        kronecker = kwargs.pop('kronecker', 1.0)
        eta_phi = kwargs.pop('eta_phi', 1e-4)
        gamma_theta = kwargs.pop('gamma_theta', 1e-5)
        gamma_phi = kwargs.pop('gamma_phi', 1e-5)
        alpha_min = kwargs.pop('alpha_min', 2.22045e-15)

        if self.size_filter == 0:
            raise RuntimeError('Filter must be initialized with at least one pair before calling search')

        alpha = max_alpha
        self._n_backtrack = 0

        opt = self.evaluate_optimality(x)
        feas = self.evaluate_fesibility(x)
        grad_opt = self.evaluate_grad_optimality(x)
        mk = grad_opt.dot(dx)
        counter = 0
        for i in range(max_iter):
            x_trial = x + alpha * dx
            opt_p = self.evaluate_optimality(x_trial)
            feas_p = self.evaluate_fesibility(x_trial)
            reject = False

            # check if is larger than max infeasibility
            if feas_p >= self._theta_max:
                reject = True

            # check if trial is in the filter
            for pair in self._filter:
                if opt_p >= pair[0] and feas_p >= pair[1]:
                    reject = True
                    break

            if not reject:

                if feas <= self._theta_min and mk < 0.0 and alpha * (-mk) ** s_phi > kronecker * feas ** s_theta:
                    if opt_p <= opt + eta_phi * alpha * mk:
                        break
                    else:
                        # armijo did not hold update alpha
                        self._n_backtrack += 1
                        alpha *= rho
                else:
                    if feas_p <= (1 - gamma_theta) * feas or opt_p <= opt - gamma_phi * feas:
                        break
                    else:
                        self._n_backtrack += 1
                        alpha *= rho
            else:
                self._n_backtrack += 1
                alpha *= rho

            if alpha < alpha_min:
                raise RuntimeError('Step is to small. Feasibility restoration needed')
            counter = i

        # update filter
        if not (feas <= self._theta_min and mk < 0.0 and alpha * (-mk) ** s_phi > kronecker * feas ** s_theta):
            for pair in self._filter[:]:
                if pair[1] >= (1-gamma_theta) * feas and pair[0] >= opt - gamma_phi * feas:
                    self._filter.remove(pair)
            self.add_to_filter(opt - gamma_phi * feas, (1-gamma_theta)*feas)

        if counter == max_iter-1:
            raise RuntimeError('Line search failed. Increase backtracking iterations.')

        return alpha



# np.finfo(float).eps