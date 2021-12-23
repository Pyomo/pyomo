#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


class FilterElement:

    def __init__(self, objective, feasible):
        self.objective = objective
        self.feasible = feasible

    def compare(self, filterElement):
        if (filterElement.objective >= self.objective
            and filterElement.feasible >= self.feasible):
            return -1
        if (filterElement.objective <= self.objective
            and filterElement.feasible <= self.feasible):
            return 1
        return 0


class Filter:
    """
    Trust region filter
    
    Based on original filter by Eason, Biegler (2016)
    """
    def __init__(self):
        self.TrustRegionFilter = []

    def addToFilter(self, filterElement):
        filtercopy = list(self.TrustRegionFilter)
        for fe in filtercopy:
            acceptableMeasure = fe.compare(filterElement)
            if (acceptableMeasure == 1):
                self.TrustRegionFilter.remove(fe)
            elif (acceptableMeasure == -1):
                return
        self.TrustRegionFilter.append(filterElement)

    def isAcceptable(self, filterElement, maximum_feasibility):
        if (filterElement.feasible > maximum_feasibility):
            return False
        for fe in self.TrustRegionFilter:
            if (fe.compare(filterElement) == -1):
                return False
        return True
