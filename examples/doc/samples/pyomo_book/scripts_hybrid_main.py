from pyomo.environ import *

def disease_mdl(discretization, disease_data):

    model = ConcreteModel()

    # attach non-Pyomo data values to the model instance.
    model.pts_LS = discretization['LSP']
    model.pts_LS_lower = discretization['LSPL']
    model.pts_LI = discretization['LIP']
    model.pts_LI_lower = discretization['LIPL']

    model.TIME = Set(ordered=True,initialize=disease_data['TIME'])
    model.BIRTHS = Param(model.TIME,initialize=disease_data['BIRTHS'])

    # more parameter and set definitions...

    model.logS = Var(model.TIME,bounds=logS_bounds_rule)
    model.logI = Var(model.TIME,bounds=logI_bounds_rule)

    # more model variables...

    model.obj = Objective(rule=obj_rule)
    model.pn_con = Constraint(model.TIME, rule=pn_rule)

    # more model constraints ...

    # automatically generate, via a function, additional
    # constraints associated with linearization and add
    # them to the model...
    linearize_exp(model.TIME,model.S,model.logS, \
                  model.pts_LS,model.pts_LS_lower)
    linearize_exp(model.TIME,model.I,model.logI, \
                  model.pts_LI,model.pts_LI_lower)

    return model

data_file = "disease_data.dat"
output_file = "global_opt.results"

# the initialize function is a utility specific to this
# example, which extracts data from the input file and
# various input arguments (not shown), and returns two
# Python dictionaries.
disease_data, discretization = initialize(data_file)

max_gap = 0.01
max_iter = 100
for i in range(max_iter):

    # define the full optimization model for this iteration.
    # data is significantly changing each iteration...
    model = disease_mdl(disease_data, discretization)

    # create and solve MIP under-estimator.
    MIP_results = solve_MIP(model, MIP_options)

    # create and solve the NLP over-estimator.
    NLP_results = solve_NLP(model, MIP_results, NLP_options)

    # load results, report status, and compute the gap/ub.
    gap, upper_bound = process_results(model,
                                       MIP_results,
                                       NLP_results,
                                       output_file)

    # use results to determine parameters for the next
    # iteration, via updates to the discretization
    points_added = refine_discretization(model,
                                         MIP_results,
                                         discretization)

    if (ub != None) and (i == 1):
        # perform solves to strengthen MIP under-estimator
        # updates the discretization
        tighten_bounds(model, MIP_options, discretization)

    if (points_added == 0) or (gap <= max_gap):
        break
