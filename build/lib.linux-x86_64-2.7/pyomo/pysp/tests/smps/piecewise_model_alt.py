from pyomo.core import *
from pyomo.pysp.annotations import (PySP_ConstraintStageAnnotation,
                                    PySP_StochasticRHSAnnotation,
                                    PySP_StochasticMatrixAnnotation,
                                    PySP_StochasticObjectiveAnnotation)

from piecewise_model import (pysp_scenario_tree_model_callback,
                             create_instance)


def pysp_instance_creation_callback(scenario_name, node_names):

    model = create_instance(scenario_name)

    #
    # SMPS Related Annotations
    #

    model.constraint_stage = PySP_ConstraintStageAnnotation()
    model.constraint_stage.declare(model.c_first_stage, 1)
    for constraint_container in \
        model.p_first_stage.component_objects(Constraint,
                                              active=True):
        model.constraint_stage.declare(constraint_container, 1)
    model.constraint_stage.declare(model.c_second_stage, 2)
    model.constraint_stage.declare(model.r_second_stage, 2)
    model.constraint_stage.declare(model.p_second_stage[1], 2)

    # The difficulty with Piecewise is that it hides the
    # structure of the underlying constraints (there may be
    # more than one). It doesn't seem possible to direct a
    # modeler on how to go about tagging specific
    # constraints.  For this reason, we allow the
    # PySP_StochasticRHS and PySP_StochasticMatrix suffixes
    # to contain entries for entire blocks, where we
    # interpret this as meaning all rhs and constraint
    # matrix entries should be treated as stochastic.
    model.stoch_rhs = PySP_StochasticRHSAnnotation()
    for constraint_data in model.p_second_stage[1].component_data_objects(Constraint,
                                                                       active=True):
        model.stoch_rhs.declare(constraint_data)
    model.stoch_rhs.declare(model.r_second_stage)
    model.stoch_matrix = PySP_StochasticMatrixAnnotation()
    # exercise more of the code by testing this with an
    # indexed block and a single block
    model.stoch_matrix.declare(model.c_second_stage, variables=[model.r])
    model.stoch_matrix.declare(model.p_second_stage[1])

    model.stoch_objective = PySP_StochasticObjectiveAnnotation()
    model.stoch_objective.declare(model.o)

    return model
