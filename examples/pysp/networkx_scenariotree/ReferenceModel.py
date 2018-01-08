#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as aml

# Creates an instance for each scenario
def pysp_instance_creation_callback(scenario_name, node_names):

    model = aml.ConcreteModel()
    model.x = aml.Var()
    model.z = aml.Var()
    model.FirstStageCost = aml.Expression(
        expr=5*(model.z**2 + (model.x-1.1)**2))
    model.SecondStageCost = aml.Expression(expr=0.0)
    model.ThirdStageCost = aml.Expression(expr=0.0)
    model.obj = aml.Objective(expr= model.FirstStageCost + \
                                    model.SecondStageCost + \
                                    model.ThirdStageCost)
    model.c = aml.ConstraintList()
    model.c.add(model.z == model.x)
    if scenario_name.startswith("u0"):
        # All scenarios under second-stage node "u0"
        model.xu0 = aml.Var()
        model.c.add(model.xu0 == model.x)
        model.SecondStageCost.expr = (model.xu0 - 1)**2

        model.y0 = aml.Var()
        model.c.add(expr= -10 <= model.y0 <= 10)
        if scenario_name == "u00":
            model.yu00 = aml.Var()
            model.c.add(model.yu00 == model.y0)
            model.ThirdStageCost.expr = (model.yu00 + 1)**2
        elif scenario_name == "u01":
            model.yu01 = aml.Var()
            model.c.add(model.yu01 == model.y0)
            model.ThirdStageCost.expr = (2*model.yu01 - 3)**2 + 1
        else:
            assert False
    elif scenario_name.startswith("u1"):
        # All scenarios under second-stage node "u1"
        model.xu1 = aml.Var()
        model.c.add(model.xu1 == model.x)
        model.SecondStageCost.expr = (model.xu1 + 1)**2

        model.y1 = aml.Var()
        model.c.add(expr= -10 <= model.y1 <= 10)
        if scenario_name == "u10":
            model.yu10 = aml.Var()
            model.c.add(model.yu10 == model.y1)
            model.ThirdStageCost.expr = (0.5*model.yu10 - 1)**2 - 1
        elif scenario_name == "u11":
            model.yu11 = aml.Var()
            model.c.add(model.yu11 == model.y1)
            model.ThirdStageCost.expr = (0.2*model.yu11)**2
        else:
            assert False
    elif scenario_name.startswith("u2"):
        # All scenarios under second-stage node "u2"
        model.xu2 = aml.Var()
        model.c.add(model.xu2 == model.x)
        model.SecondStageCost.expr = (model.xu2 - 0.5)**2

        model.y2 = aml.Var()
        model.c.add(expr= -10 <= model.y2 <= 10)
        if scenario_name == "u20":
            model.yu20 = aml.Var()
            model.c.add(model.yu20 == model.y2)
            model.ThirdStageCost.expr = (0.1*model.yu20 - 3)**2 + 2
        else:
            assert False
    else:
        assert False

    return model
