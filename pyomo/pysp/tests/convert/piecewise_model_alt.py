#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import Constraint
from pyomo.pysp.annotations import (StochasticConstraintBoundsAnnotation,
                                    StochasticConstraintBodyAnnotation,
                                    StochasticObjectiveAnnotation)

from piecewise_model import create_instance

def pysp_instance_creation_callback(scenario_name, node_names):

    model = create_instance(scenario_name)

    #
    # SMPS Related Annotations
    #
    model.stoch_rhs = StochasticConstraintBoundsAnnotation()
    for con in model.p_second_stage[1].component_data_objects(
            Constraint,
            active=True):
        model.stoch_rhs.declare(con)
    model.stoch_rhs.declare(model.r_second_stage)
    model.stoch_matrix = StochasticConstraintBodyAnnotation()
    # exercise more of the code by testing this with an
    # indexed block and a single block
    model.stoch_matrix.declare(model.c_second_stage, variables=[model.r])
    model.stoch_matrix.declare(model.p_second_stage[1])

    model.stoch_objective = StochasticObjectiveAnnotation()
    model.stoch_objective.declare(model.o)

    return model
