import os

from pyomo.common.fileutils import this_file_dir
from pyomo.environ import *
from pyomo.gdp import *

# from pyomo.opt import SolverFactory
# from pyomo.core.base import Transformation


''' Layout optimization for screening systems in waste paper recovery:
Problem from http://www.minlp.org/library/problem/index.php?i=263&lib=GDP
This problem is a design problem: When waste paper is recovered, a separator 
screen is used to separate the fiber from the stickies. Different reject rates 
can be set for different pieces of equipment to separate out the output with 
more fiber and the output with more stickies. The overall equipment layout 
affects the output purity and the flow rates. (We want more fiber and fewer 
stickies!)'''


def build_model():
    model = AbstractModel()


    model.BigM = Suffix(direction=Suffix.LOCAL)
    model.BigM[None] = 1000

    DATFILE = "stickies1.dat"

    #######################
    #Sets
    #######################

    # J
    model.Components = Set()
    # fiber
    model.GoodComponents = Set()
    # stickies
    model.BadComponents = Set()
    # N: total nodes in the system
    model.Nodes = Set()
    # S: possibe screens
    model.Screens = Set()

    def screen_node_filter(model, s, n):
        return s != n
    model.ScreenNodePairs = Set(initialize=model.Screens * model.Nodes, dimen=2,
                                filter=screen_node_filter)

    def screen_filter(model, s, sprime):
        return s != sprime
    model.ScreenPairs = Set(initialize = model.Screens * model.Screens, dimen=2,
                            filter=screen_filter)

    ######################
    # Parameters
    ######################

    # exponent coefficient for cost in screen s (alpha(s))
    model.ExpScreenCostCoeff = Param(model.Screens)
    # beta(s, j)
    model.AcceptanceFactor = Param(model.Screens, model.Components)
    # C_s^1
    model.ScreenCostCoeff1 = Param(model.Screens)
    # C_s^2
    model.ScreenCostCoeff2 = Param(model.Screens, default=0)
    # max percentage inlet stickies accepted in total flow (C_{st}^{up}, q(kb))
    model.AcceptedLeftover = Param(model.BadComponents)
    # F_j^0, m_src(k)
    model.InitialComponentFlow = Param(model.Components)
    # m_src_lo(k)
    model.InitialComponentFlowLB = Param(model.Components, default=0)

    # constants for objective function (W^1, W^2, W^3)
    model.FiberWeight = Param()
    model.StickiesWeight = Param()
    model.CostWeight = Param()


    ## Bounds on variables

    # F_s^{in, lo} and F_s^{in, up} (f_in_up(s), f_in_lo(s))
    def flow_ub_rule(model, s):
        return sum(model.InitialComponentFlow[k] for k in model.Components)
    model.ScreenFlowLB = Param(model.Screens)
    model.ScreenFlowUB = Param(model.Screens, initialize=flow_ub_rule)

    # m_in_lo(ss, k): lower bound of individual flow into nodes.
    model.InletComponentFlowLB = Param(model.Components, model.Nodes, default=0)
    def component_flow_ub_rule(model, k, n):
        return model.InitialComponentFlow[k]
    # m_in_up(ss, k)
    model.InletComponentFlowUB = Param(model.Components, model.Nodes,
                                       initialize=component_flow_ub_rule)

    # r_lo(s)
    model.RejectRateLB = Param(model.Screens)
    # r_up(s)
    model.RejectRateUB = Param(model.Screens)

    # m_rej_lo(s, k)
    model.RejectedComponentFlowLB = Param(model.Components, model.Screens,
                                          default=0)
    def rejected_component_flow_bound(model, k, s):
        return model.InitialComponentFlow[k]*(model.RejectRateUB[s]**\
                                     model.AcceptanceFactor[s, k])
    # m_rej_up(s, k)
    model.RejectedComponentFlowUB = Param(model.Components, model.Screens,
                                          initialize=rejected_component_flow_bound)

    # m_acc_lo(s, k): lower bound of accepted individual flow
    model.AcceptedComponentFlowLB = Param(model.Components, model.Screens,
                                          default=0)
    def accepted_component_flow_bound(model, k, s):
        return model.InitialComponentFlow[k]*(1 - model.RejectRateLB[s]**\
                                     model.AcceptanceFactor[s, k])
    # m_acc_up(s, k)
    model.AcceptedComponentFlowUB = Param(model.Components, model.Screens,
                                          initialize=accepted_component_flow_bound)

    ######################
    # Variables
    ######################

    # c_s, C(s), cost of selecting screen
    model.screenCost = Var(model.Screens, within=NonNegativeReals)#, bounds=get_screen_cost_bounds)

    # total inlet flow into screen s (f_s, F_IN(s))
    # NOTE: the upper bound is enforced globally. The lower bound is enforced in
    # the first disjunction (to match GAMS)
    def get_inlet_flow_bounds(model, s):
        return (0, model.ScreenFlowUB[s])
    model.inletScreenFlow = Var(model.Screens, within=NonNegativeReals,
                                bounds=get_inlet_flow_bounds)

    # inlet flow of component j into node n, (f_{n,j}^I, M_IN)
    def get_inlet_component_flow_bounds(model, j, n):
        return (model.InletComponentFlowLB[j, n], model.InletComponentFlowUB[j, n])
    model.inletComponentFlow = Var(model.Components, model.Nodes,
                                   within=NonNegativeReals,
                                   bounds=get_inlet_component_flow_bounds)

    # accepted flow of component j from screen s (f_{s, j}^A)
    def get_accepted_component_flow_bounds(model, j, s):
        return (model.AcceptedComponentFlowLB[j, s],
                model.AcceptedComponentFlowUB[j, s])
    model.acceptedComponentFlow = Var(model.Components, model.Screens,
                                      within=NonNegativeReals,
                                      bounds=get_accepted_component_flow_bounds)
    # rejected flow of component j from screen s (f_{s,j}^R)
    def rej_component_flow_bounds(model, k, s):
        return (model.RejectedComponentFlowLB[k, s],
                model.RejectedComponentFlowUB[k, s])
    model.rejectedComponentFlow = Var(model.Components, model.Screens,
                                      within=NonNegativeReals,
                                      bounds=rej_component_flow_bounds)

    # accepted flow of component j from screen s to node n (m_{s,n,j}^A)
    def get_accepted_node_flow_bounds(model, j, s, n):
        return (0, model.AcceptedComponentFlowUB[j, s])
    model.acceptedNodeFlow = Var(model.Components, model.Screens, model.Nodes,
                                 within=NonNegativeReals,
                                 bounds=get_accepted_node_flow_bounds)

    # rejected flow of component j from screen s to node n (m_{s,n,j}^R)
    def get_rejected_node_flow_bounds(model, j, s, n):
        return (0, model.RejectedComponentFlowUB[j, s])
    model.rejectedNodeFlow = Var(model.Components, model.Screens, model.Nodes,
                                 within=NonNegativeReals,
                                 bounds=get_rejected_node_flow_bounds)

    # flow of component j from source to node n (m_{s,j}^0)
    def get_src_flow_bounds(model, j, n):
        return (0, model.InitialComponentFlow[j])
    model.flowFromSource = Var(model.Components, model.Nodes,
                               within=NonNegativeReals)

    # reject rate of screen s (r_s)
    def get_rej_rate_bounds(model, s):
        return (model.RejectRateLB[s], model.RejectRateUB[s])
    model.rejectRate = Var(model.Screens, within=NonNegativeReals,
                           bounds=get_rej_rate_bounds)


    ######################
    # Objective
    ######################

    def calc_cost_rule(model):
        lostFiberCost = model.FiberWeight * sum(model.inletComponentFlow[j,'SNK'] \
                                               for j in model.GoodComponents)
        stickiesCost = model.StickiesWeight * sum(model.inletComponentFlow[j,'PRD']\
                                                  for j in model.BadComponents)
        screenCost = model.CostWeight * sum(model.screenCost[s] \
                                            for s in model.Screens)
        return lostFiberCost + stickiesCost + screenCost
    model.min_cost = Objective(rule=calc_cost_rule)


    ######################
    # Constraints
    ######################

    def stickies_bound_rule(model, j):
        return sum(model.inletComponentFlow[j,'PRD'] for j in model.BadComponents) \
            <= model.AcceptedLeftover[j] * model.InitialComponentFlow[j]
    model.stickies_bound = Constraint(model.BadComponents, rule=stickies_bound_rule)

    def inlet_flow_rule(model, s, j):
        return model.inletComponentFlow[j,s] == model.acceptedComponentFlow[j,s] + \
            model.rejectedComponentFlow[j, s]
    model.inlet_flow = Constraint(model.Screens, model.Components,
                                  rule=inlet_flow_rule)

    def total_inlet_flow_rule(model, s):
        return model.inletScreenFlow[s] == sum(model.inletComponentFlow[j, s] \
                                               for j in model.Components)
    model.total_inlet_flow = Constraint(model.Screens, rule=total_inlet_flow_rule)

    def inlet_flow_balance_rule(model, n, j):
        return model.inletComponentFlow[j, n] == model.flowFromSource[j, n] + \
            sum(model.acceptedNodeFlow[j, s, n] + model.rejectedNodeFlow[j, s, n] \
                for s in model.Screens if s != n)
    model.inlet_flow_balance = Constraint(model.Nodes, model.Components,
                                          rule=inlet_flow_balance_rule)

    def source_flow_rule(model, j):
        return model.InitialComponentFlow[j] == sum(model.flowFromSource[j, n] \
                                           for n in model.Nodes)
    model.source_flow = Constraint(model.Components, rule=source_flow_rule)

    #################
    ## Disjunctions
    #################

    def screen_disjunct_rule(disjunct, selectScreen, s):
        model = disjunct.model()
        def rejected_flow_rule(disjunct, j):
            return model.rejectedComponentFlow[j,s] == \
                model.inletComponentFlow[j,s]* \
                (model.rejectRate[s]**model.AcceptanceFactor[s, j])

        if selectScreen:
            disjunct.inlet_flow_bounds = Constraint(expr=model.ScreenFlowLB[s] <= \
                                                    model.inletScreenFlow[s])# <= \
                                                    #model.ScreenFlowUB[s])
            disjunct.rejected_flow = Constraint(model.Components,
                                                rule=rejected_flow_rule)
            disjunct.screen_cost = Constraint(expr=model.screenCost[s] == \
                                              model.ScreenCostCoeff1[s]* \
                                              (model.inletScreenFlow[s]** \
                                               model.ExpScreenCostCoeff[s]) + \
                                              model.ScreenCostCoeff2[s]* \
                                              (1 - model.rejectRate[s]))
        else:
            disjunct.no_flow = Constraint(expr=model.inletScreenFlow[s] == 0)
            disjunct.no_cost = Constraint(expr=model.screenCost[s] == 0)
    model.screen_selection_disjunct = Disjunct([0,1], model.Screens,
                                               rule=screen_disjunct_rule)

    def screen_disjunction_rule(model, s):
        return [model.screen_selection_disjunct[selectScreen, s] \
                for selectScreen in [0,1]]
    model.screen_disjunction = Disjunction(model.Screens,
                                           rule=screen_disjunction_rule)


    def accepted_flow_disjunct_rule(disjunct, s, n, acceptFlow):
        model = disjunct.model()
        def flow_balance_rule(disjunct, j):
            return model.acceptedNodeFlow[j, s, n] == \
                model.acceptedComponentFlow[j, s]
        def no_flow_rule(disjunct, j):
            return model.acceptedNodeFlow[j, s, n] == 0

        if acceptFlow:
            disjunct.flow_balance = Constraint(model.Components,
                                               rule=flow_balance_rule)
        else:
            disjunct.no_flow = Constraint(model.Components, rule=no_flow_rule)
    model.flow_acceptance_disjunct = Disjunct(model.ScreenNodePairs, [0,1],
                                              rule=accepted_flow_disjunct_rule)

    def flow_acceptance_disjunction_rule(model, s, n):
        return [model.flow_acceptance_disjunct[s, n, acceptFlow] \
                for acceptFlow in [0,1]]
    model.flow_acceptance_disjunction = Disjunction(model.ScreenNodePairs,
                                            rule=flow_acceptance_disjunction_rule)


    def rejected_flow_disjunct_rule(disjunct, s, n, rejectFlow):
        model = disjunct.model()
        def flow_balance_rule(disjunct, j):
            return model.rejectedNodeFlow[j, s, n] == \
                model.rejectedComponentFlow[j, s]
        def no_reject_rule(disjunct, j):
            return model.rejectedNodeFlow[j, s, n] == 0

        if rejectFlow:
            disjunct.flow_balance = Constraint(model.Components,
                                               rule=flow_balance_rule)
        else:
            disjunct.no_reject = Constraint(model.Components, rule=no_reject_rule)
    model.flow_rejection_disjunct = Disjunct(model.ScreenNodePairs, [0,1],
                                             rule=rejected_flow_disjunct_rule)

    def rejected_flow_disjunction_rule(model, s, n):
        return [model.flow_rejection_disjunct[s, n, rejectFlow] \
                for rejectFlow in [0,1]]
    model.flow_rejection_disjunction = Disjunction(model.ScreenNodePairs,
                                                rule=rejected_flow_disjunction_rule)


    def flow_from_source_disjunct_rule(disjunct, n):
        model = disjunct.model()
        def sourceFlow_balance_rule1(disjunct, j):
            # this doesn't match the formulation, but it matches GAMS:
            return model.flowFromSource[j, n] >= model.InitialComponentFlowLB[j]
            # this would be the formulation version:
            #return model.flowFromSource[j, n] == model.InitialComponentFlow[j]
        def sourceFlow_balance_rule2(disjunct, j):
            return model.flowFromSource[j, n] <= model.InitialComponentFlow[j]
        def no_sourceFlow_rule(disjunct, j, nprime):
            return model.flowFromSource[j, nprime] == 0

        disjunct.flow_balance1 = Constraint(model.Components,
                                            rule=sourceFlow_balance_rule1)
        disjunct.flow_balance2 = Constraint(model.Components,
                                            rule=sourceFlow_balance_rule2)
        disjunct.no_flow = Constraint(model.Components, model.Nodes - [n],
                                      rule=no_sourceFlow_rule)
    model.flow_from_source_disjunct = Disjunct(model.Nodes,
                                               rule=flow_from_source_disjunct_rule)

    def flow_from_source_disjunction_rule(model):
        return [model.flow_from_source_disjunct[n] for n in model.Nodes]
    model.flow_from_source_disjunction = Disjunction(
        rule=flow_from_source_disjunction_rule)


    ######################
    # Boolean Constraints
    ######################

    # These are the GAMS versions of the logical constraints, which is not
    # what appears in the formulation:
    def log1_rule(model, s):
        return model.screen_selection_disjunct[1, s].indicator_var == \
            sum(model.flow_acceptance_disjunct[s, n, 1].indicator_var \
                for n in model.Nodes if s != n)
    model.log1 = Constraint(model.Screens, rule=log1_rule)

    def log2_rule(model, s):
        return model.screen_selection_disjunct[1, s].indicator_var == \
            sum(model.flow_rejection_disjunct[s, n, 1].indicator_var \
                for n in model.Nodes if s != n)
    model.log2 = Constraint(model.Screens, rule=log2_rule)

    def log3_rule(model, s):
        return model.screen_selection_disjunct[1, s].indicator_var >= \
            sum(model.flow_acceptance_disjunct[s, sprime, 1].indicator_var \
                for sprime in model.Screens if s != sprime)
    model.log3 = Constraint(model.Screens, rule=log3_rule)

    def log4_rule(model, s):
        return model.screen_selection_disjunct[1, s].indicator_var >= \
            sum(model.flow_rejection_disjunct[s, sprime, 1].indicator_var \
                for sprime in model.Screens if s != sprime)
    model.log4 = Constraint(model.Screens, rule=log4_rule)

    def log6_rule(model, s, sprime):
        return model.flow_acceptance_disjunct[s, sprime, 1].indicator_var + \
            model.flow_acceptance_disjunct[sprime, s, 1].indicator_var <= 1
    model.log6 = Constraint(model.ScreenPairs, rule=log6_rule)

    def log7_rule(model, s, sprime):
        return model.flow_rejection_disjunct[s, sprime, 1].indicator_var + \
            model.flow_rejection_disjunct[sprime, s, 1].indicator_var <= 1
    model.log7 = Constraint(model.ScreenPairs, rule=log7_rule)

    def log8_rule(model, s, n):
        return model.flow_acceptance_disjunct[s, n, 1].indicator_var + \
            model.flow_rejection_disjunct[s, n, 1].indicator_var <= 1
    model.log8 = Constraint(model.ScreenNodePairs, rule=log8_rule)

    def log9_rule(model, s, sprime):
        return model.flow_acceptance_disjunct[s, sprime, 1].indicator_var + \
            model.flow_rejection_disjunct[sprime, s, 1].indicator_var <= 1
    model.log9 = Constraint(model.ScreenPairs, rule=log9_rule)

    # These are the above logical constraints implemented correctly (I think)
    # However, this doesn't match what is actually coded in gams and makes the
    # model infeasible with the data from gams.

    # YA_{s,n} v YR_{s,n} implies Y_s
    # def flow_existence_rule1(model, s, n):
    #     return model.screen_selection_disjunct[1, s].indicator_var >= \
    #         model.flow_acceptance_disjunct[s, n, 1].indicator_var
    # model.flow_existence1 = Constraint(model.ScreenNodePairs,
    #                                    rule=flow_existence_rule1)

    # def flow_existence_rule2(model, s, n):
    #     return model.screen_selection_disjunct[1, s].indicator_var >= \
    #         model.flow_rejection_disjunct[s, n, 1].indicator_var
    # model.flow_existence2 = Constraint(model.ScreenNodePairs,
    #                                    rule=flow_existence_rule2)


    # # YA_{s,s'} v YR_{s',s} implies Y_s
    # def screen_flow_existence_rule1(model, s, sprime):
    #     return model.screen_selection_disjunct[1, s].indicator_var >= \
    #         model.flow_acceptance_disjunct[s, sprime, 1].indicator_var
    # model.screen_flow_existence1 = Constraint(model.ScreenPairs,
    #                                           rule=screen_flow_existence_rule1)

    # def screen_flow_existence_rule2(model, s, sprime):
    #     return model.screen_selection_disjunct[1,s].indicator_var >= \
    #         model.flow_rejection_disjunct[sprime, s, 1].indicator_var
    # model.screen_flow_existence2 = Constraint(model.ScreenPairs,
    #                                           rule=screen_flow_existence_rule2)


    # # YA_{s', s} XOR YA_{s, s'}
    # def accept_rule1(model, s, sprime):
    #     return 1 <= model.flow_acceptance_disjunct[s, sprime, 1].indicator_var + \
    #         model.flow_acceptance_disjunct[sprime, s, 1].indicator_var
    # model.accept1 = Constraint(model.ScreenPairs, rule=accept_rule1)

    # def accept_rule2(model, s, sprime):
    #     return 1 >= model.flow_acceptance_disjunct[s, sprime, 1].indicator_var - \
    #         model.flow_acceptance_disjunct[sprime, s, 1].indicator_var
    # model.accept2 = Constraint(model.ScreenPairs, rule=accept_rule2)

    # def accept_rule3(model, s, sprime):
    #     return 1 >= model.flow_acceptance_disjunct[sprime, s, 1].indicator_var - \
    #         model.flow_acceptance_disjunct[s, sprime, 1].indicator_var
    # model.accept3 = Constraint(model.ScreenPairs, rule=accept_rule3)

    # def accept_rule4(model, s, sprime):
    #     return 1 <= 2 - model.flow_acceptance_disjunct[sprime,s,1].indicator_var - \
    #         model.flow_acceptance_disjunct[s, sprime, 1].indicator_var
    # model.accept4 = Constraint(model.ScreenPairs, rule=accept_rule4)


    # # YR_{s', s} XOR YR_{s, s'}
    # def reject_rule1(model, s, sprime):
    #     return 1 <= model.flow_rejection_disjunct[s, sprime, 1].indicator_var + \
    #         model.flow_rejection_disjunct[sprime, s, 1].indicator_var
    # model.reject1 = Constraint(model.ScreenPairs, rule=reject_rule1)

    # def reject_rule2(model, s, sprime):
    #     return 1 >= model.flow_rejection_disjunct[s, sprime, 1].indicator_var - \
    #         model.flow_rejection_disjunct[sprime, s, 1].indicator_var
    # model.reject2 = Constraint(model.ScreenPairs, rule=reject_rule2)

    # def reject_rule3(model, s, sprime):
    #     return 1 >= model.flow_rejection_disjunct[sprime, s, 1].indicator_var - \
    #         model.flow_rejection_disjunct[s, sprime, 1].indicator_var
    # model.reject3 = Constraint(model.ScreenPairs, rule=reject_rule3)

    # def reject_rule4(model, s, sprime):
    #     return 1 <= 2 - model.flow_rejection_disjunct[sprime,s,1].indicator_var - \
    #         model.flow_rejection_disjunct[s, sprime, 1].indicator_var
    # model.reject4 = Constraint(model.ScreenPairs, rule=reject_rule4)


    # # YA_{s,n} XOR YR_{s,n}
    # def accept_or_reject_rule1(model, s, n):
    #     return 1 <= model.flow_acceptance_disjunct[s, n, 1].indicator_var + \
    #         model.flow_rejection_disjunct[s, n, 1].indicator_var
    # model.accept_or_reject1 = Constraint(model.ScreenNodePairs,
    #                                      rule=accept_or_reject_rule1)

    # def accept_or_reject_rule2(model, s, n):
    #     return 1 >= model.flow_acceptance_disjunct[s, n, 1].indicator_var - \
    #         model.flow_rejection_disjunct[s, n, 1].indicator_var
    # model.accept_or_reject2 = Constraint(model.ScreenNodePairs,
    #                                      rule=accept_or_reject_rule2)

    # def accept_or_reject_rule3(model, s, n):
    #     return 1 >= model.flow_rejection_disjunct[s, n, 1].indicator_var - \
    #         model.flow_acceptance_disjunct[s, n, 1].indicator_var
    # model.accept_or_reject3 = Constraint(model.ScreenNodePairs,
    #                                      rule=accept_or_reject_rule3)

    # def accept_or_reject_rule4(model, s, n):
    #     return 1 <= 2 - model.flow_acceptance_disjunct[s, n, 1].indicator_var - \
    #         model.flow_rejection_disjunct[s, n, 1].indicator_var
    # model.accept_or_reject4 = Constraint(model.ScreenNodePairs,
    #                                      rule=accept_or_reject_rule4)

    instance = model.create_instance(os.path.join(this_file_dir(), DATFILE))

    # fix the variables they fix in GAMS
    for s in instance.Screens:
        instance.flow_acceptance_disjunct[s,'SNK',1].indicator_var.fix(0)
        instance.flow_rejection_disjunct[s,'PRD',1].indicator_var.fix(0)

    ##################################################################################
    ## for validation: Fix all the indicator variables to see if we get same objective
    ## value (250.956)
    ##################################################################################

    # instance.screen_selection_disjunct[1,'S1'].indicator_var.fix(1)
    # instance.screen_selection_disjunct[1,'S2'].indicator_var.fix(0)
    # instance.screen_selection_disjunct[1,'S3'].indicator_var.fix(0)
    # instance.screen_selection_disjunct[1,'S4'].indicator_var.fix(0)
    # instance.screen_selection_disjunct[1,'S5'].indicator_var.fix(0)
    # instance.screen_selection_disjunct[1,'S6'].indicator_var.fix(0)


    # instance.flow_acceptance_disjunct['S1','S2',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S1','S3',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S1','S4',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S1','S5',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S1','S6',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S1','PRD',1].indicator_var.fix(1)
    # # 'SNK' is already fixed correctly in the loop above that is "in" the model
    # instance.flow_acceptance_disjunct['S2','S1',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S2','S3',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S2','S4',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S2','S5',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S2','S6',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S2','PRD',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S3','S1',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S3','S2',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S3','S4',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S3','S5',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S3','S6',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S3','PRD',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S4','S1',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S4','S2',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S4','S3',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S4','S5',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S4','S6',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S4','PRD',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S5','S1',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S5','S2',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S5','S3',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S5','S4',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S5','S6',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S5','PRD',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S6','S1',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S6','S2',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S6','S3',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S6','S4',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S6','S5',1].indicator_var.fix(0)
    # instance.flow_acceptance_disjunct['S6','PRD',1].indicator_var.fix(0)

    # instance.flow_rejection_disjunct['S1','S2',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S1','S3',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S1','S4',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S1','S5',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S1','S6',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S1','SNK',1].indicator_var.fix(1)
    # # 'SNK' is already fixed correctly in the loop above that is "in" the model
    # instance.flow_rejection_disjunct['S2','S1',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S2','S3',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S2','S4',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S2','S5',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S2','S6',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S2','SNK',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S3','S1',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S3','S2',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S3','S4',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S3','S5',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S3','S6',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S3','SNK',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S4','S1',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S4','S2',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S4','S3',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S4','S5',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S4','S6',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S4','SNK',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S5','S1',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S5','S2',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S5','S3',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S5','S4',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S5','S6',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S5','SNK',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S6','S1',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S6','S2',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S6','S3',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S6','S4',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S6','S5',1].indicator_var.fix(0)
    # instance.flow_rejection_disjunct['S6','SNK',1].indicator_var.fix(0)

    # instance.flow_from_source_disjunct['S1'].indicator_var.fix(1)
    # instance.flow_from_source_disjunct['S2'].indicator_var.fix(0)
    # instance.flow_from_source_disjunct['S3'].indicator_var.fix(0)
    # instance.flow_from_source_disjunct['S4'].indicator_var.fix(0)
    # instance.flow_from_source_disjunct['S5'].indicator_var.fix(0)
    # instance.flow_from_source_disjunct['S6'].indicator_var.fix(0)
    # instance.flow_from_source_disjunct['PRD'].indicator_var.fix(0)
    # instance.flow_from_source_disjunct['SNK'].indicator_var.fix(0)

    return instance


if __name__ == '__main__':
    instance = build_model()

    # bigm transformation!
    bigM = TransformationFactory('gdp.bigm')
    bigM.apply_to(instance)

    # solve the model
    opt = SolverFactory('baron')
    opt.options["MaxTime"] = 32400
    results = opt.solve(instance, tee=True)

    instance.solutions.store_to(results)
    print(results)
