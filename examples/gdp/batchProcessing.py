#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction

'''Problem from http://www.minlp.org/library/problem/index.php?i=172&lib=GDP
We are minimizing the cost of a design of a plant with parallel processing units and storage tanks
in between. We decide the number and volume of units, and the volume and location of the storage
tanks. The problem is convexified and has a nonlinear objective and global constraints

NOTE: When I refer to 'gams' in the comments, that is Batch101006_BM.gms for now. It's confusing
because the _opt file is different (It has hard-coded bigM parameters so that each constraint 
has the "optimal" bigM).'''


def build_model():
    model = pyo.AbstractModel()

    # TODO: it looks like they set a bigM for each j. Which I need to look up how to do...
    model.BigM = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    model.BigM[None] = 1000

    ## Constants from GAMS
    StorageTankSizeFactor = (
        2 * 5
    )  # btw, I know 2*5 is 10... I don't know why it's written this way in GAMS?
    StorageTankSizeFactorByProd = 3
    MinFlow = -pyo.log(10000)
    VolumeLB = pyo.log(300)
    VolumeUB = pyo.log(3500)
    StorageTankSizeLB = pyo.log(100)
    StorageTankSizeUB = pyo.log(15000)
    UnitsInPhaseUB = pyo.log(6)
    UnitsOutOfPhaseUB = pyo.log(6)
    # TODO: YOU ARE HERE. YOU HAVEN'T ACTUALLY MADE THESE THE BOUNDS YET, NOR HAVE YOU FIGURED OUT WHOSE
    # BOUNDS THEY ARE. AND THERE ARE MORE IN GAMS.

    ##########
    # Sets
    ##########

    model.PRODUCTS = pyo.Set()
    model.STAGES = pyo.Set(ordered=True)
    model.PARALLELUNITS = pyo.Set(ordered=True)

    # TODO: this seems like an over-complicated way to accomplish this task...
    def filter_out_last(model, j):
        return j != model.STAGES.last()

    model.STAGESExceptLast = pyo.Set(initialize=model.STAGES, filter=filter_out_last)

    # TODO: these aren't in the formulation??
    # model.STORAGE_TANKS = pyo.Set()

    ###############
    # Parameters
    ###############

    model.HorizonTime = pyo.Param()
    model.Alpha1 = pyo.Param()
    model.Alpha2 = pyo.Param()
    model.Beta1 = pyo.Param()
    model.Beta2 = pyo.Param()

    model.ProductionAmount = pyo.Param(model.PRODUCTS)
    model.ProductSizeFactor = pyo.Param(model.PRODUCTS, model.STAGES)
    model.ProcessingTime = pyo.Param(model.PRODUCTS, model.STAGES)

    # These are hard-coded in the GAMS file, hence the defaults
    model.StorageTankSizeFactor = pyo.Param(model.STAGES, default=StorageTankSizeFactor)
    model.StorageTankSizeFactorByProd = pyo.Param(
        model.PRODUCTS, model.STAGES, default=StorageTankSizeFactorByProd
    )

    # TODO: bonmin wasn't happy and I think it might have something to do with this?
    # or maybe issues with convexity or a lack thereof... I don't know yet.
    # I made PRODUCTS ordered so I could do this... Is that bad? And it does index
    # from 1, right?
    def get_log_coeffs(model, k):
        return pyo.log(model.PARALLELUNITS.ord(k))

    model.LogCoeffs = pyo.Param(model.PARALLELUNITS, initialize=get_log_coeffs)

    # bounds
    model.volumeLB = pyo.Param(model.STAGES, default=VolumeLB)
    model.volumeUB = pyo.Param(model.STAGES, default=VolumeUB)
    model.storageTankSizeLB = pyo.Param(model.STAGES, default=StorageTankSizeLB)
    model.storageTankSizeUB = pyo.Param(model.STAGES, default=StorageTankSizeUB)
    model.unitsInPhaseUB = pyo.Param(model.STAGES, default=UnitsInPhaseUB)
    model.unitsOutOfPhaseUB = pyo.Param(model.STAGES, default=UnitsOutOfPhaseUB)

    ################
    # Variables
    ################

    # TODO: right now these match the formulation. There are more in GAMS...

    # unit size of stage j
    # model.volume = pyo.Var(model.STAGES)
    # # TODO: GAMS has a batch size indexed just by products that isn't in the formulation... I'm going
    # # to try to avoid it for the moment...
    # # batch size of product i at stage j
    # model.batchSize = pyo.Var(model.PRODUCTS, model.STAGES)
    # # TODO: this is different in GAMS... They index by stages too?
    # # cycle time of product i divided by batch size of product i
    # model.cycleTime = pyo.Var(model.PRODUCTS)
    # # number of units in parallel out-of-phase (or in phase) at stage j
    # model.unitsOutOfPhase = pyo.Var(model.STAGES)
    # model.unitsInPhase = pyo.Var(model.STAGES)
    # # TODO: what are we going to do as a boundary condition here? For that last stage?
    # # size of intermediate storage tank between stage j and j+1
    # model.storageTankSize = pyo.Var(model.STAGES)

    # variables for convexified problem
    # TODO: I am beginning to think these are my only variables actually.
    # GAMS never un-logs them, I don't think. And I think the GAMs ones
    # must be the log ones.
    def get_volume_bounds(model, j):
        return (model.volumeLB[j], model.volumeUB[j])

    model.volume_log = pyo.Var(model.STAGES, bounds=get_volume_bounds)
    model.batchSize_log = pyo.Var(model.PRODUCTS, model.STAGES)
    model.cycleTime_log = pyo.Var(model.PRODUCTS)

    def get_unitsOutOfPhase_bounds(model, j):
        return (0, model.unitsOutOfPhaseUB[j])

    model.unitsOutOfPhase_log = pyo.Var(model.STAGES, bounds=get_unitsOutOfPhase_bounds)

    def get_unitsInPhase_bounds(model, j):
        return (0, model.unitsInPhaseUB[j])

    model.unitsInPhase_log = pyo.Var(model.STAGES, bounds=get_unitsInPhase_bounds)

    def get_storageTankSize_bounds(model, j):
        return (model.storageTankSizeLB[j], model.storageTankSizeUB[j])

    # TODO: these bounds make it infeasible...
    model.storageTankSize_log = pyo.Var(model.STAGES, bounds=get_storageTankSize_bounds)

    # binary variables for deciding number of parallel units in and out of phase
    model.outOfPhase = pyo.Var(model.STAGES, model.PARALLELUNITS, within=pyo.Binary)
    model.inPhase = pyo.Var(model.STAGES, model.PARALLELUNITS, within=pyo.Binary)

    ###############
    # Objective
    ###############

    def get_cost_rule(model):
        return model.Alpha1 * sum(
            pyo.exp(
                model.unitsInPhase_log[j]
                + model.unitsOutOfPhase_log[j]
                + model.Beta1 * model.volume_log[j]
            )
            for j in model.STAGES
        ) + model.Alpha2 * sum(
            pyo.exp(model.Beta2 * model.storageTankSize_log[j])
            for j in model.STAGESExceptLast
        )

    model.min_cost = pyo.Objective(rule=get_cost_rule)

    ##############
    # Constraints
    ##############

    def processing_capacity_rule(model, j, i):
        return (
            model.volume_log[j]
            >= pyo.log(model.ProductSizeFactor[i, j])
            + model.batchSize_log[i, j]
            - model.unitsInPhase_log[j]
        )

    model.processing_capacity = pyo.Constraint(
        model.STAGES, model.PRODUCTS, rule=processing_capacity_rule
    )

    def processing_time_rule(model, j, i):
        return (
            model.cycleTime_log[i]
            >= pyo.log(model.ProcessingTime[i, j])
            - model.batchSize_log[i, j]
            - model.unitsOutOfPhase_log[j]
        )

    model.processing_time = pyo.Constraint(
        model.STAGES, model.PRODUCTS, rule=processing_time_rule
    )

    def finish_in_time_rule(model):
        return model.HorizonTime >= sum(
            model.ProductionAmount[i] * pyo.exp(model.cycleTime_log[i])
            for i in model.PRODUCTS
        )

    model.finish_in_time = pyo.Constraint(rule=finish_in_time_rule)

    ###############
    # Disjunctions
    ###############

    def storage_tank_selection_disjunct_rule(disjunct, selectStorageTank, j):
        model = disjunct.model()

        def volume_stage_j_rule(disjunct, i):
            return (
                model.storageTankSize_log[j]
                >= pyo.log(model.StorageTankSizeFactor[j]) + model.batchSize_log[i, j]
            )

        def volume_stage_jPlus1_rule(disjunct, i):
            return (
                model.storageTankSize_log[j]
                >= pyo.log(model.StorageTankSizeFactor[j])
                + model.batchSize_log[i, j + 1]
            )

        def batch_size_rule(disjunct, i):
            return pyo.inequality(
                -pyo.log(model.StorageTankSizeFactorByProd[i, j]),
                model.batchSize_log[i, j] - model.batchSize_log[i, j + 1],
                pyo.log(model.StorageTankSizeFactorByProd[i, j]),
            )

        def no_batch_rule(disjunct, i):
            return model.batchSize_log[i, j] - model.batchSize_log[i, j + 1] == 0

        if selectStorageTank:
            disjunct.volume_stage_j = pyo.Constraint(
                model.PRODUCTS, rule=volume_stage_j_rule
            )
            disjunct.volume_stage_jPlus1 = pyo.Constraint(
                model.PRODUCTS, rule=volume_stage_jPlus1_rule
            )
            disjunct.batch_size = pyo.Constraint(model.PRODUCTS, rule=batch_size_rule)
        else:
            # The formulation says 0, but GAMS has this constant.
            # 04/04: Francisco says volume should be free:
            # disjunct.no_volume = pyo.Constraint(expr=model.storageTankSize_log[j] == MinFlow)
            disjunct.no_batch = pyo.Constraint(model.PRODUCTS, rule=no_batch_rule)

    model.storage_tank_selection_disjunct = Disjunct(
        [0, 1], model.STAGESExceptLast, rule=storage_tank_selection_disjunct_rule
    )

    def select_storage_tanks_rule(model, j):
        return [
            model.storage_tank_selection_disjunct[selectTank, j]
            for selectTank in [0, 1]
        ]

    model.select_storage_tanks = Disjunction(
        model.STAGESExceptLast, rule=select_storage_tanks_rule
    )

    # though this is a disjunction in the GAMs model, it is more efficiently formulated this way:
    # TODO: what on earth is k?
    def units_out_of_phase_rule(model, j):
        return model.unitsOutOfPhase_log[j] == sum(
            model.LogCoeffs[k] * model.outOfPhase[j, k] for k in model.PARALLELUNITS
        )

    model.units_out_of_phase = pyo.Constraint(
        model.STAGES, rule=units_out_of_phase_rule
    )

    def units_in_phase_rule(model, j):
        return model.unitsInPhase_log[j] == sum(
            model.LogCoeffs[k] * model.inPhase[j, k] for k in model.PARALLELUNITS
        )

    model.units_in_phase = pyo.Constraint(model.STAGES, rule=units_in_phase_rule)

    # and since I didn't do the disjunction as a disjunction, we need the XORs:
    def units_out_of_phase_xor_rule(model, j):
        return sum(model.outOfPhase[j, k] for k in model.PARALLELUNITS) == 1

    model.units_out_of_phase_xor = pyo.Constraint(
        model.STAGES, rule=units_out_of_phase_xor_rule
    )

    def units_in_phase_xor_rule(model, j):
        return sum(model.inPhase[j, k] for k in model.PARALLELUNITS) == 1

    model.units_in_phase_xor = pyo.Constraint(
        model.STAGES, rule=units_in_phase_xor_rule
    )

    return model


if __name__ == "__main__":
    m = build_model().create_instance('batchProcessing1.dat')
    pyo.TransformationFactory('gdp.bigm').apply_to(m)
    pyo.SolverFactory('gams').solve(
        m, solver='baron', tee=True, add_options=['option optcr=1e-6;']
    )
    m.min_cost.display()
