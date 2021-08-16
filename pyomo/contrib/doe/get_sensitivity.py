#import idaes
import pandas as pd
import numpy as np
from scipy import sparse
import pyomo.contrib.parmest.parmest as parmest
from pyomo.environ import *
from pyomo.opt import SolverFactory
import shutil
import logging


def get_sensitivity(model, theta_names, tee=True, solver_options=None):
    '''
    Parameters
    ----------
    model: Pyomo ConcreteModel
    theta_names: list of strings
        List of Var names
    tee: bool, optional
        Indicates that ef solver output should be teed
    solver_options: dict, optional
        Provides options to the solver (also the name of an attribute)
    
    Returns
    -------
    gradient_f: numpy.ndarray
        gradient vector of the objective function with respect to all decision variables at the optimal solution
    gradient_c: numpy.ndarray
        gradient vector of the constraints with respect to all decision variables at the optimal solution
        Each row contains column number, row number, and value
        If no constraint exists, return []
    line_dic: dict
        column numbers of the theta_names in the model. Index starts from 1
    Raises
    ------
    RuntimeError
        When ipopt or kaug or dotsens is not available
    Exception
        When ipopt fails 
    '''
    #Create the solver plugin using the ASL interface
    
    ipopt = SolverFactory('ipopt',solver_io='nl')
    ipopt.options['linear_solver']='ma57'
    if solver_options is not None:
        ipopt.options = solver_options
    kaug = SolverFactory('k_aug',solver_io='nl')
    dotsens = SolverFactory('dot_sens',solver_io='nl')
    if not ipopt.available(False):
        raise RuntimeError('ipopt is not available')
    if not kaug.available(False):
        raise RuntimeError('k_aug is not available')
    if not dotsens.available(False):
        raise RuntimeError('dotsens is not available')

    # Declare Suffixes
    model.dual = Suffix(direction = Suffix.IMPORT)
    model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
    model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

    # K_AUG SUFFIXES
    model.dof_v = Suffix(direction=Suffix.EXPORT)  #: SUFFIX FOR K_AUG
    model.rh_name = Suffix(direction=Suffix.IMPORT)  #: SUFFIX FOR K_AUG AS WELL
    kaug.options["print_kkt"] = ""
    results = ipopt.solve(model,tee=tee)

    # Rasie Exception if ipopt fails 
    if (results.solver.status == pyomo.opt.SolverStatus.warning):
        raise Exception(results.solver.Message)

    for o in model.component_objects(Objective, active=True):
        f_mean = value(o)
    model.ipopt_zL_in.update(model.ipopt_zL_out)
    model.ipopt_zU_in.update(model.ipopt_zU_out)
    #: run k_aug
    kaug.solve(model, tee=tee)  #: always call k_aug AFTER ipopt.
    model.write('col_row.nl', format='nl', io_options={'symbolic_solver_labels':True})
    # get the column numbers of theta
    line_dic = {}
    for v in theta_names:
        line_dic[v] = line_num('col_row.col', v)
    # load gradient of the objective function
    gradient_f = np.loadtxt("./GJH/gradient_f_print.txt")
    # load gradient of all constraints (sparse)
    # If no constraint exists, return []
    num_constraints = len(list(model.component_data_objects(Constraint,
                                                            active=True,
                                                            descend_into=True)))
    if num_constraints > 0 :
        gradient_c = np.loadtxt("./GJH/kkt_print.txt")
        gradient_c = reorganize_kkt(gradient_c)
    else:
        gradient_c = np.array([])

    # remove all generated files
    #shutil.move("col_row.nl", "./GJH/")
    #shutil.move("col_row.col", "./GJH/")
    #shutil.move("col_row.row", "./GJH/")
    #shutil.rmtree('GJH', ignore_errors=True)
    return gradient_f, gradient_c, line_dic


def reorganize_kkt(gradient):
    '''
    Reorganize information in kkt_print.txt to Jacobian matrix 
    '''
    jaco_kkt = np.zeros((27,4))
    
    gra_dict={}

    for c in range(np.shape(gradient)[0]):
        l = []
        l.append(gradient[c][0])
        l.append(gradient[c][1])
        gra_dict[tuple(l)] = gradient[c][2]
    
    for k in range(32,40):
        jaco_kkt[k-31][0] = gra_dict[tuple([1,k])] 
        jaco_kkt[k-31][2] = gra_dict[tuple([3,k])] 
        
    for q in range(40,48):
        for x in range(4):
            jaco_kkt[q-30][x] = gra_dict[tuple([x+1,q])]
    
    for l in range(48,56):
        for t in range(4):
            jaco_kkt[l-29][t] = gra_dict[tuple([t+1,l])]
    return jaco_kkt


def get_sensitivity_original(model, theta_names, tee=True, solver_options=None):
    '''
    Parameters
    ----------
    model: Pyomo ConcreteModel
    theta_names: list of strings
        List of Var names
    tee: bool, optional
        Indicates that ef solver output should be teed
    solver_options: dict, optional
        Provides options to the solver (also the name of an attribute)
    
    Returns
    -------
    gradient_f: numpy.ndarray
        gradient vector of the objective function with respect to all decision variables at the optimal solution
    gradient_c: numpy.ndarray
        gradient vector of the constraints with respect to all decision variables at the optimal solution
        Each row contains column number, row number, and value
        If no constraint exists, return []
    line_dic: dict
        column numbers of the theta_names in the model. Index starts from 1
    Raises
    ------
    RuntimeError
        When ipopt or kaug or dotsens is not available
    Exception
        When ipopt fails 
    '''
    #Create the solver plugin using the ASL interface
    
    ipopt = SolverFactory('ipopt',solver_io='nl')
    ipopt.options['linear_solver']='ma57'
    if solver_options is not None:
        ipopt.options = solver_options
    kaug = SolverFactory('k_aug',solver_io='nl')
    dotsens = SolverFactory('dot_sens',solver_io='nl')
    if not ipopt.available(False):
        raise RuntimeError('ipopt is not available')
    if not kaug.available(False):
        raise RuntimeError('k_aug is not available')
    if not dotsens.available(False):
        raise RuntimeError('dotsens is not available')

    # Declare Suffixes
    model.dual = Suffix(direction = Suffix.IMPORT)
    model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
    model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

    # K_AUG SUFFIXES
    model.dof_v = Suffix(direction=Suffix.EXPORT)  #: SUFFIX FOR K_AUG
    model.rh_name = Suffix(direction=Suffix.IMPORT)  #: SUFFIX FOR K_AUG AS WELL
    kaug.options["print_kkt"] = ""
    results = ipopt.solve(model,tee=tee)

    # Rasie Exception if ipopt fails 
    if (results.solver.status == pyomo.opt.SolverStatus.warning):
        raise Exception(results.solver.Message)

    for o in model.component_objects(Objective, active=True):
        f_mean = value(o)
    model.ipopt_zL_in.update(model.ipopt_zL_out)
    model.ipopt_zU_in.update(model.ipopt_zU_out)
    #: run k_aug
    kaug.solve(model, tee=tee)  #: always call k_aug AFTER ipopt.
    model.write('col_row.nl', format='nl', io_options={'symbolic_solver_labels':True})
    # get the column numbers of theta
    line_dic = {}
    for v in theta_names:
        line_dic[v] = line_num('col_row.col', v)
    # load gradient of the objective function
    gradient_f = np.loadtxt("./GJH/gradient_f_print.txt")
    # load gradient of all constraints (sparse)
    # If no constraint exists, return []
    num_constraints = len(list(model.component_data_objects(Constraint,
                                                            active=True,
                                                            descend_into=True)))
    if num_constraints > 0 :
        gradient_c = np.loadtxt("./GJH/A_print.txt")
    else:
        gradient_c = np.array([])

    # remove all generated files
    #shutil.move("col_row.nl", "./GJH/")
    #shutil.move("col_row.col", "./GJH/")
    #shutil.move("col_row.row", "./GJH/")
    #shutil.rmtree('GJH', ignore_errors=True)
    return gradient_f, gradient_c, line_dic

def line_num(file_name, target):
    """This function returns the line number contains 'target' in the file_name
    Parameters
    ----------
    file_name: string
        file name includes information of variabe order (col_row.col)
    target: string   
        variable name to check  
    Returns
    -------
    count: int
        line number of target in the file_name
        
    Raises
    ------
    Exception
        When col_row.col doesnot include target
    """
    with open(file_name) as f:
        count = int(1)
        for line in f:
            if line.strip() == target:
                return int(count)
            count += 1
    return count
    #raise Exception("col_row.col should includes target")

def clean_variable_name(theta_names):
    """This eunction removes all ' and spaces in theta_names.
       Note: The  current theta_est(calc_cov=True) of parmest in Pyomo
       doesn't allow ' and spaces in the variable names    
    Parameters
    ----------
    theta_names: list of strings
        List of Var names
    
    Returns
    -------
    theta_names_out: list of strings
        List of Var names after removing  all ' and spaces
    var_dic: dict
       dictionary with keys converted theta_names and values origianl theta_names 
    """

    # Variable names cannot have "'" for parmest_class.theta_est(calc_cov=True)
    # Save original variables name in to var_dic
    # Remove all "'" and " " in theta_names
    var_dic = {}
    theta_names_out = []
    clean = False
    for i in range(len(theta_names)):
        if "'" in theta_names[i] or " " in theta_names[i] :
            logger.warning(theta_names[i] + " includes ' or space.")
            clean = True
        theta_tmp = theta_names[i].replace("'", '')
        theta_tmp = theta_tmp.replace(" ", '')
        theta_names_out.append(theta_tmp)
        var_dic[theta_tmp] = theta_names[i]
    if clean:
        logger.warning("All ' and spaces in theta_names are removed.")
    
    return theta_names_out, var_dic

