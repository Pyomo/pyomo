
class InertiaCorrectionParams(object):

    def __init__(self):

        self.delta_w_min = 1e-20
        self.delta_w_zero = 1e-4
        self.delta_w_max = 1e40

        self.delta_a_hat = 1e-10
        self.kappa_w_plus = 8.0
        self.kappa_w_minus = 1.0/3.0
        self.kappa_w_plus_hat = 100.0
        self.kappa_a = 1.0/4.0
        self.delta_w_last = 0.0

        self.delta_w = 0.0
        self.delta_a = 0.0
        self._recorded_delta_w = 0.0

    def reset(self):
        self.delta_w_min = 1e-20
        self.delta_w_zero = 1e-4
        self.delta_w_max = 1e40

        self.delta_a_hat = 1e-8
        self.kappa_w_plus = 8.0
        self.kappa_w_minus = 1.0 / 3.0
        self.kappa_w_plus_hat = 100.0
        self.kappa_a = 1.0 / 4.0
        self.delta_w_last = 0.0

        self.delta_w = 0.0
        self.delta_a = 0.0
        self._recorded_delta_w = 0.0

    def ibr1(self, solve_status):
        if solve_status == 0:
            self.delta_w = 0.0
            self.delta_a = 0.0
            return True
        self._ibr2(solve_status)
        self._ibr3()
        return False

    def _ibr2(self, solve_status):
        if solve_status == 1:
            self.delta_a = self.delta_a_hat
        else:
            self.delta_a = 0.0

    def _ibr3(self):
        if self.delta_w_last == 0:
            self.delta_w = self.delta_w_zero
        else:
            self.delta_w = max(self.delta_w_min,
                               self.kappa_w_minus * self.delta_w_last)

    def ibr4(self, solve_status):
        if solve_status == 0:
            self.delta_w_last = self.delta_w
            self.delta_w = 0.0
            self.delta_a = 0.0
            return True
        self._ibr5()
        self._ibr6()
        return False

    def _ibr5(self):
        if self.delta_w_last == 0:
            self.delta_w *= self.kappa_w_plus_hat
        else:
            self.delta_w *= self.kappa_w_plus

    def _ibr6(self):
        if self.delta_w > self.delta_w_max:
            raise RuntimeError('Quiting: Problem could not be regularized. May need restoration')
