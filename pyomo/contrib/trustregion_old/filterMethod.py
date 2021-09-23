
class FilterElement:

    def __init__(self, objective, infeasible):
        self.objective = objective
        self.infeasible = infeasible

    def compare(self, x):
        if (x.objective >= self.objective and x.infeasible >= self.infeasible):
            return -1
        if (x.objective <= self.objective and x.infeasible <= self.infeasible):
            return 1
        return 0


class Filter:
    filteR = []

    def addToFilter(self, x):
        filtercopy = list(self.filteR)
        for i in filtercopy:
            if (i.compare(x) == 1):
                self.filteR.remove(i)
            elif (i.compare(x) == -1):
                return
        self.filteR.append(x)

    def checkAcceptable(self, x, theta_max):
        if (x.infeasible > theta_max):
            return False
        for i in self.filteR:
            if (i.compare(x) == -1):
                return False
        return True
