"""
The classes defined here are meant to facilitate the direct use of gurobipy through pyomo.
"""
import pyomo
import pyomo.environ as pe
import gurobipy


def gurobi_vtype_from_domain(domain):
    domain = type(domain)
    if domain == pyomo.core.base.set_types.RealSet:
        vtype = gurobipy.GRB.CONTINUOUS
    elif domain == pyomo.core.base.set_types.BooleanSet:
        vtype = gurobipy.GRB.BINARY
    else:
        raise ValueError('Variable type is not recognized for {0}: {1}'.format(value, domain))
    return vtype

class GurobiModelContainer(object):
    """
    A container for gurobipy models

    Attributes
    ----------
    model: gurobi.Model
    """

    def __init__(self):
        self.model = gurobipy.Model()


class GurobiModel(pe.ConcreteModel):
    """
    A class to facilitate the use of gurobipy

    Attributes
    ----------
    gmc: GurobiModelContainer
    """

    def __init__(self):
        super(GurobiModel, self).__init__()
        self.gmc = GurobiModelContainer()

    def __setattr__(self, name, value):
        super(GurobiModel, self).__setattr__(name, value)

        if isinstance(value, pe.Var):
            if value.is_indexed():
                index_set = list(value.index_set())
                domain = value[index_set[0]].domain
                for i in index_set:
                    assert value[i].domain == domain
                vtype = gurobi_vtype_from_domain(domain)
                lb = {i:value[i].lb if value[i].lb is not None else -gurobipy.GRB.INFINITY for i in index_set}
                ub = {i:value[i].ub if value[i].lb is not None else gurobipy.GRB.INFINITY for i in index_set}
                gurobipy_var = self.gmc.model.addVars(index_set, lb=lb, ub=ub, vtype=vtype, name=name)
                setattr(self.gmc, name, gurobipy_var)

            else:
                domain = value.domain
                vtype = gurobi_vtype_from_domain(domain)
                lb = value.lb
                ub = value.ub
                if lb is None:
                    lb = -gurobipy.GRB.INFINITY
                if ub is None:
                    ub = gurobipy.GRB.INFINITY
                gurobipy_var = self.gmc.model.addVar(lb=lb, ub=ub, vtype=vtype, name=name)
                setattr(self.gmc, name, gurobipy_var)