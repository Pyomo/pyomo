# ______________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License
# ______________________________________________________________________________
from pyomo.environ import (
        Param,
        Var,
        Block,
        ComponentMap,
        Objective,
        Constraint,
        ConstraintList,
        Suffix,
        value,
        ComponentUID,
        )

from pyomo.core.base.misc import sorted_robust
from pyomo.core.expr.current import ExpressionReplacementVisitor

from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated
from pyomo.opt import SolverFactory, SolverStatus
import logging
import os
import shutil
import numpy as np

logger = logging.getLogger('pyomo.contrib.sensitivity_toolbox')

@deprecated("The sipopt function has been deprecated. Use the sensitivity_calculation() "
            "function with method='sipopt' to access this functionality.",
            logger='pyomo.contrib.sensitivity_toolbox',
            version='TBD')
def sipopt(instance, paramSubList, perturbList,
           cloneModel=True, tee=False, keepfiles=False,
           streamSoln=False):
    m = sensitivity_calculation('sipopt', instance, paramSubList, perturbList,
         cloneModel, tee, keepfiles, solver_options=None)

    return m

@deprecated("The kaug function has been deprecated. Use the sensitivity_calculation() "
            "function with method='kaug' to access this functionality.", 
            logger='pyomo.contrib.sensitivity_toolbox',
            version='TBD')
def kaug(instance, paramSubList, perturbList,
         cloneModel=True, tee=False, keepfiles=False, solver_options=None,
         streamSoln=False):
    m = sensitivity_calculation('kaug', instance, paramSubList, perturbList,
         cloneModel, tee, keepfiles, solver_options)

    return m

_SIPOPT_SUFFIXES = {
        'sens_state_0': Suffix.EXPORT,
        # ^ Not sure what this suffix does -RBP
        'sens_state_1': Suffix.EXPORT,
        'sens_state_value_1': Suffix.EXPORT,
        'sens_init_constr': Suffix.EXPORT,

        'sens_sol_state_1': Suffix.IMPORT,
        'sens_sol_state_1_z_L': Suffix.IMPORT,
        'sens_sol_state_1_z_U': Suffix.IMPORT,
        }

_K_AUG_SUFFIXES = {
        'ipopt_zL_out': Suffix.IMPORT,
        'ipopt_zU_out': Suffix.IMPORT,
        'ipopt_zL_in': Suffix.EXPORT,
        'ipopt_zU_in': Suffix.EXPORT,
        'dual': Suffix.IMPORT_EXPORT,
        'dcdp': Suffix.EXPORT,
        'DeltaP': Suffix.EXPORT,
        }

def _add_sensitivity_suffixes(block):
    suffix_dict = {}
    suffix_dict.update(_SIPOPT_SUFFIXES)
    suffix_dict.update(_K_AUG_SUFFIXES)
    for name, direction in suffix_dict.items():
        if block.component(name) is None:
            # Only add suffix if it doesn't already exist.
            # If something of this name does already exist, just
            # assume it is the proper suffix and move on.
            block.add_component(name, Suffix(direction=direction))

class _NotAnIndex(object):
    pass

def _generate_component_items(components):
    if type(components) not in {list, tuple}:
        components = (components,)
    for comp in components:
        if comp.is_indexed():
            for idx in sorted_robust(comp):
                yield idx, comp[idx]
        else:
            yield _NotAnIndex, comp

def sensitivity_calculation(method, instance, paramList, perturbList,
         cloneModel=True, tee=False, keepfiles=False, solver_options=None):
    sens = SensitivityInterface(instance, clone_model=cloneModel)
    sens.setup_sensitivity(paramList)

    m = sens.model_instance

    if method == 'kaug':
        kaug = SolverFactory('k_aug', solver_io='nl')
        dotsens = SolverFactory('dot_sens', solver_io='nl')
        ipopt = SolverFactory('ipopt', solver_io='nl')

        ipopt.solve(m, tee=tee)
        m.ipopt_zL_in.update(m.ipopt_zL_out)  #: important!
        m.ipopt_zU_in.update(m.ipopt_zU_out)  #: important!    

        kaug.options['dsdp_mode'] = ""  #: sensitivity mode!
        kaug.solve(m, tee=tee)
        m.write('col_row.nl', format='nl', io_options={'symbolic_solver_labels':True})

    sens.perturb_parameters(perturbList)

    if method == 'sipopt':
        ipopt_sens = SolverFactory('ipopt_sens', solver_io='nl')
        ipopt_sens.options['run_sens'] = 'yes'

        # Send the model to the ipopt_sens and collect the solution
        results = ipopt_sens.solve(m, keepfiles=keepfiles, tee=tee)

    elif method == 'kaug':
        dotsens.options["dsdp_mode"] = ""
        dotsens.solve(m, tee=tee)
        try:
            os.makedirs("dsdp")
        except FileExistsError:
            # directory already exists
            pass
        try:
            shutil.move("dsdp_in_.in","./dsdp/")
            shutil.move("col_row.nl","./dsdp/")
            shutil.move("col_row.col","./dsdp/")
            shutil.move("col_row.row","./dsdp/")
            shutil.move("conorder.txt","./dsdp/")
            shutil.move("delta_p.out","./dsdp/")
            shutil.move("dot_out.out","./dsdp/")
            shutil.move("timings_dot_driver_dsdp.txt", "./dsdp/")
            shutil.move("timings_k_aug_dsdp.txt", "./dsdp/")
        except OSError:
            pass

    return m

def get_dsdp(model, theta_names, theta, var_dic={},tee=False, solver_options=None):
    """This function calculates gradient vector of the (decision variables, parameters)
        with respect to the paramerters (theta_names).
    e.g) min f:  p1*x1+ p2*(x2^2) + p1*p2
         s.t  c1: x1 + x2 = p1
              c2: x2 + x3 = p2
              0 <= x1, x2, x3 <= 10
              p1 = 10
              p2 = 5
    the function retuns dx/dp and dp/dp, and colum orders.
    Parameters
    ----------
    model: Pyomo ConcreteModel
        model should includes an objective function
    theta_names: list of strings
        List of Var names
    theta: dict
        Estimated parameters e.g) from parmest
    tee: bool, optional
        Indicates that ef solver output should be teed
    solver_options: dict, optional
        Provides options to the solver (also the name of an attribute)
    var_dic: dictionary
        If any original variable contains "'", need an auxiliary dictionary 
        with keys theta_names without "'", values with "'".
        e.g) var_dic: {'fs.properties.tau[benzene,toluene]': "fs.properties.tau['benzene','toluene']",
                       'fs.properties.tau[toluene,benzene]': "fs.properties.tau['toluene','benzene']"}

    Returns
    -------
    dsdp_dic: dict
        gradient vector of the (decision variables, parameters) with respect to paramerters (=theta_name).
        e.g) dict = {'d(x1)/d(p1)', 'd(x2)/d(p1)', 'd(p1)/d(p1)', 'd(p2)/d(p1)', 'd(x3)/d(p1)', 
                     'd(x1)/d(p2)', 'd(x2)/d(p2)', 'd(p1)/d(p2)', 'd(p2)/d(p2)', 'd(x3)/d(p2)'},
    col: list
        list of variable names
        e.g) col = ['x1', 'x2', 'p1', 'p2', 'x3'].
    """
    m = model.clone()
    original_Param = []
    perturbed_Param = []
    m.extra = ConstraintList()
    kk = 0
    if var_dic == {}:
        for i in theta_names:
            var_dic[i] = i
    for v in theta_names:
        v_tmp = str(kk)
        original_param_object = Param(initialize=theta[v], mutable=True)
        perturbed_param_object = Param(initialize=theta[v])
        m.add_component("original_"+v_tmp, original_param_object)
        m.add_component("perturbed_"+v_tmp, perturbed_param_object)
        m.extra.add(eval('m.'+var_dic[v]) - eval('m.original_'+v_tmp) == 0 )
        original_Param.append(original_param_object)
        perturbed_Param.append(perturbed_param_object)
        kk = kk + 1
    m_kaug_dsdp = sensitivity_calculation('kaug',m,original_Param,perturbed_Param, tee)

    try:
        with open ("./dsdp/col_row.col", "r") as myfile:
            col = myfile.read().splitlines()
        dsdp = np.loadtxt("./dsdp/dsdp_in_.in")
    except Exception as e:
        print('File not found.')

    dsdp = dsdp.reshape((len(theta_names), int(len(dsdp)/len(theta_names))))
    dsdp = dsdp[:len(theta_names), :len(col)]
    dsdp_dic = {}
    for i in range(len(theta_names)):
        for j in range(len(col)):
            if SensitivityInterface.get_default_block_name() not in col[j]:
                dsdp_dic["d("+col[j] +")/d("+theta_names[i]+")"] =  -dsdp[i, j]
    try:
        shutil.rmtree('dsdp', ignore_errors=True)
    except OSError:
        pass
    col = [i for i in col if SensitivityInterface.get_default_block_name() not in i]
    return dsdp_dic, col

def get_dfds_dcds(model, theta_names, tee=False, solver_options=None):
    """This function calculates gradient vector of the objective function 
       and constraints with respect to the variables in theta_names.
    e.g) min f:  p1*x1+ p2*(x2^2) + p1*p2
         s.t  c1: x1 + x2 = p1
              c2: x2 + x3 = p2
              0 <= x1, x2, x3 <= 10
              p1 = 10
              p2 = 5
    - Variables = (x1, x2, x3, p1, p2)
    - Fix p1 and p2 with estimated values
    - The function provides gradient vector at the optimal solution
      gradient vector of the objective function, 
      'd(f)/d(x1)', 'd(f)/d(x2)', 'd(f)/d(x3)', 'd(f)/d(p1)', 'd(f)/d(p2)',
      gradient vector of the constraints, 
      'd(c1)/d(x1), 'd(c1)/d(x2)', 'd(c1)/d(p1)', 'd(c2)/d(x2)', 'd(c2)/d(p2)', 'd(c2)/d(x3)'.

    Parameters
    ----------
    model: Pyomo ConcreteModel
        model should includes an objective function 
    theta_names: list of strings
        List of Var names
    tee: bool, optional
        Indicates that ef solver output should be teed
    solver_options: dict, optional
        Provides options to the solver (also the name of an attribute)
    
    Returns
    -------
    gradient_f: numpy.ndarray
        gradient vector of the objective function with respect to the (decision variables, parameters) at the optimal solution
    gradient_f_dic: dic
        gradient_f with variable name as key 
        e.g) dic = {'d(f)/d(x1)': 10.0, 'd(f)/d(x2)': 50.0, 'd(f)/d(p1)': 15.0, 'd(f)/d(p2)': 35.0}
    gradient_c: numpy.ndarray
        gradient vector of the constraints with respect to the (decision variables, parameters) at the optimal solution
        Each row contains column number, row number, and value
        If no constraint exists, return []
    gradient_c: dic
        gradient_c with constraint number and variable name as key 
        e.g) dic = {'d(c1)/d(x1)': 1.0, 'd(c1)/d(p1)': -1.0, 'd(c2)/d(x2)': 1.0, 'd(c2)/d(p2)': -1.0}
        Only non-zero gradients are included.
    line_dic: dict
        column numbers of the theta_names in the model. Index starts from 1

    Raises
    ------
    RuntimeError
        When ipopt or kaug or dotsens is not available
    Exception
        When ipopt fails 
    """
    #Create the solver plugin using the ASL interface
    ipopt = SolverFactory('ipopt',solver_io='nl')
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
    _add_sensitivity_suffixes(model)

    # K_AUG SUFFIXES
    model.dof_v = Suffix(direction=Suffix.EXPORT)  #: SUFFIX FOR K_AUG
    model.rh_name = Suffix(direction=Suffix.IMPORT)  #: SUFFIX FOR K_AUG AS WELL
    kaug.options["print_kkt"] = ""
    results = ipopt.solve(model,tee=tee)

    # Rasie Exception if ipopt fails 
    if (results.solver.status == SolverStatus.warning):
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
    try:
        for v in theta_names:
            line_dic[v] = line_num('col_row.col', v)
        # load gradient of the objective function
        gradient_f = np.loadtxt("./GJH/gradient_f_print.txt")
        with open ("col_row.col", "r") as myfile:
            col = myfile.read().splitlines()
    except Exception as e:
        print('File not found.')
        raise e
    gradient_f_dic = {}
    for i in range(len(col)):
        gradient_f_dic["d(f)/d("+col[i]+")"] = gradient_f[i]
    # load gradient of all constraints (sparse)
    # If no constraint exists, return []
    num_constraints = len(list(model.component_data_objects(Constraint,
                                                            active=True,
                                                            descend_into=True)))
    if num_constraints > 0 :
        try:
            gradient_c = np.loadtxt("./GJH/A_print.txt")
        except Exception as e:
            print('./GJH/A_print.txt not found.')
        gradient_c = np.array([i for i in gradient_c if not np.isclose(i[2],0)])
        row_number, col_number = np.shape(gradient_c)
        gradient_c_dic = {}
        for i in range(row_number):
            gradient_c_dic["d(c"+ str(int(gradient_c[i,1]))+")/d("+col[int(gradient_c[i,0]-1)]+")"] = gradient_c[i,2]
    else:
        gradient_c = np.array([])
        gradient_c_dic = {}
    # remove all generated files
    shutil.move("col_row.nl", "./GJH/")
    shutil.move("col_row.col", "./GJH/")
    shutil.move("col_row.row", "./GJH/")
    shutil.rmtree('GJH', ignore_errors=True)
    return gradient_f,gradient_f_dic, gradient_c,gradient_c_dic, line_dic

def line_num(file_name, target):
    """This function returns the line number contains 'target' in the file_name.
    This function identities constraints that have variables in theta_names.

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
        When col_row.col does not include target
    """
    with open(file_name) as f:
        count = int(1)
        for line in f:
            if line.strip() == target:
                return int(count)
            count += 1
    raise Exception(file_name + " does not include "+target)

class SensitivityInterface(object):

    def __init__(self, instance, clone_model=True):
        """ Constructor clones model if necessary and attaches
        to this object.
        """
        self._original_model = instance

        if clone_model:
            # Note that we are not "cloning" the user's parameters
            # or perturbations.
            self.model_instance = instance.clone()
        else:
            self.model_instance = instance

    @classmethod
    def get_default_block_name(self):
        return '_SENSITIVITY_TOOLBOX_DATA'

    @staticmethod
    def get_default_var_name(name):
        #return '_'.join(('sens_var', name))
        return name

    @staticmethod
    def get_default_param_name(name):
        #return '_'.join(('sens_param', name))
        return name

    def _process_param_list(self, paramList):
        # We need to translate the components in paramList into
        # components in our possibly cloned model.
        orig = self._original_model
        instance = self.model_instance
        if orig is not instance:
            paramList = list(
                ComponentUID(param, context=orig).find_component_on(instance)
                for param in paramList
                )
        return paramList

    def _add_data_block(self, existing_block=None):
        # If a sensitivity block already exists, and we have not done
        # any expression replacement, we delete the old block, re-fix the
        # sensitivity variables, and start again.
        #
        # Don't do this in the constructor as we could want to call
        # the constructor once, then perform multiple sensitivity
        # calculations with the same model instance.
        if existing_block is not None:
            if (hasattr(existing_block, '_has_replaced_expressions') and
                    not existing_block._has_replaced_expressions):
                for var, _, _, _ in existing_block._sens_data_list:
                    # Re-fix variables that the previous block was
                    # treating as parameters.
                    var.fix()
                self.model_instance.del_component(existing_block)
            else:
                msg = ("Re-using sensitivity interface is not supported "
                        "when calculating sensitivity for mutable parameters. "
                        "Used fixed vars instead if you want to do this."
                        )
                raise RuntimeError(msg)

        # Add a block to keep track of model components necessary for this
        # sensitivity calculation.
        block = Block()
        self.model_instance.add_component(self.get_default_block_name(), block)
        self.block = block

        # If the user tells us they will perturb a set of parameters, we will
        # need to replace these parameters in the user's model's constraints.
        # This affects what we can do with the model later, so we add a flag.
        block._has_replaced_expressions = False

        # This is the main data structure for keeping track of "sensitivity
        # vars" and their associated params. It will be a list of tuples of
        # (vardata, paramdata, list_index, comp_index)
        # where:
        #     vardata is the "sensitivity variable" data object,
        #     paramdata is the associated parameter,
        #     list_index is its index in the user-provided list, and
        #     comp_index is its index in the component provided by the user.
        block._sens_data_list = []

        # This will hold the user-provided list of
        # variables and/or parameters to perturb
        block._paramList = None

        # This will hold any constraints where we have replaced
        # parameters with vairables.
        block.constList = ConstraintList()

        return block

    def _add_sensitivity_data(self, param_list):
        block = self.block
        sens_data_list = block._sens_data_list
        for i, comp in enumerate(param_list):
            if comp.ctype is Param:
                parent = comp.parent_component()
                if not parent.mutable:
                    raise ValueError(
                            "Parameters within paramList must be mutable. "
                            "Got %s, which is not mutable." % comp.name
                            )
                # Add a Var:
                if comp.is_indexed():
                    d = {k: value(comp[k]) for k in comp.index_set()}
                    var = Var(comp.index_set(), initialize=d)
                else:
                    d = value(comp)
                    var = Var(initialize=d)
                name = self.get_default_var_name(parent.local_name)
                name = unique_component_name(block, name)
                block.add_component(name, var)

                if comp.is_indexed():
                    sens_data_list.extend(
                            (var[idx], param, i, idx)
                            for idx, param in _generate_component_items(comp)
                            )
                else:
                    sens_data_list.append((var, comp, i, _NotAnIndex))

            elif comp.ctype is Var:
                parent = comp.parent_component()
                for _, data in _generate_component_items(comp):
                    if not data.fixed:
                        raise ValueError(
                                "Specified \"parameter\" variables must be "
                                "fixed. Got %s, which is not fixed."
                                % comp.name
                                )
                # Add a Param:
                if comp.is_indexed():
                    d = {k: value(comp[k]) for k in comp.index_set()}
                    param = Param(comp.index_set(), mutable=True, initialize=d)
                else:
                    d = value(comp)
                    param = Param(mutable=True, initialize=d)
                name = self.get_default_param_name(parent.local_name)
                name = unique_component_name(block, name)
                block.add_component(name, param)

                if comp.is_indexed():
                    sens_data_list.extend(
                            (var, param[idx], i, idx)
                            for idx, var in _generate_component_items(comp)
                            )
                else:
                    sens_data_list.append((comp, param, i, _NotAnIndex))

    def _replace_parameters_in_constraints(self, variableSubMap):
        instance = self.model_instance
        block = self.block
        # Visitor that we will use to replace user-provided parameters
        # in the objective and the constraints.
        param_replacer = ExpressionReplacementVisitor(
                substitute=variableSubMap,
                remove_named_expressions=True,
                )
        # TODO: Flag to ExpressionReplacementVisitor to only replace
        # named expressions if a node has been replaced within that
        # expression.

        new_old_comp_map = ComponentMap()

        # clone Objective, add to Block, and update any Expressions
        for obj in list(instance.component_data_objects(Objective,
                                                active=True,
                                                descend_into=True)):
            tempName = unique_component_name(block, obj.local_name)
            new_expr = param_replacer.dfs_postorder_stack(obj.expr)
            block.add_component(tempName, Objective(expr=new_expr))
            new_old_comp_map[block.component(tempName)] = obj
            obj.deactivate()

        # clone Constraints, add to Block, and update any Expressions
        #
        # Unfortunate that this deactivates and replaces constraints
        # even if they don't contain the parameters.
        # 
        old_con_list = list(instance.component_data_objects(Constraint,
            active=True, descend_into=True))
        last_idx = 0
        for con in old_con_list:
            if (con.equality or con.lower is None or con.upper is None):
                new_expr = param_replacer.dfs_postorder_stack(con.expr)
                block.constList.add(expr=new_expr)
                last_idx += 1
                new_old_comp_map[block.constList[last_idx]] = con
            else:
                # Constraint must be a ranged inequality, break into
                # separate constraints
                new_body = param_replacer.dfs_postorder_stack(con.body)
                new_lower = param_replacer.dfs_postorder_stack(con.lower)
                new_upper = param_replacer.dfs_postorder_stack(con.upper)

                # Add constraint for lower bound
                block.constList.add(expr=(new_lower <= new_body))
                last_idx += 1
                new_old_comp_map[block.constList[last_idx]] = con

                # Add constraint for upper bound
                block.constList.add(expr=(new_body <= new_upper))
                last_idx += 1
                new_old_comp_map[block.constList[last_idx]] = con
            con.deactivate()

        return new_old_comp_map

    def setup_sensitivity(self, paramList):
        """
        """
        instance = self.model_instance
        paramList = self._process_param_list(paramList)

        existing_block = instance.component(self.get_default_block_name())
        block = self._add_data_block(existing_block=existing_block)
        block._sens_data_list = []
        block._paramList = paramList

        self._add_sensitivity_data(paramList)
        sens_data_list = block._sens_data_list

        for var, _, _, _ in sens_data_list:
            # This unfixes all variables, not just those the user added.
            var.unfix()

        # Map used to replace user-provided parameters.
        variableSubMap = dict((id(param), var)
                for var, param, list_idx, _ in sens_data_list
                if paramList[list_idx].ctype is Param)

        if variableSubMap:
            # We now replace the provided parameters in the user's
            # expressions. Only do this if we have to, i.e. the
            # user provided some parameters rather than all vars.
            self._replace_parameters_in_constraints(variableSubMap)

            # Assume that we just replaced some params
            block._has_replaced_expressions = True

        block.paramConst = ConstraintList()
        for var, param, _, _ in sens_data_list:
            #block.paramConst.add(param - var == 0)
            block.paramConst.add(var - param == 0)

        # Declare Suffixes
        _add_sensitivity_suffixes(instance)

        for i, (var, _, _, _) in enumerate(sens_data_list):
            idx = i + 1
            con = block.paramConst[idx]

            # sipopt
            instance.sens_state_0[var] = idx
            instance.sens_state_1[var] = idx
            instance.sens_init_constr[con] = idx

            # k_aug
            instance.dcdp[con] = idx


    def perturb_parameters(self, perturbList): 
        """
        """
        # Note that entries of perturbList need not be components
        # of the cloned model. All we need are the values.
        instance = self.model_instance
        sens_data_list = self.block._sens_data_list
        paramConst = self.block.paramConst

        if len(self.block._paramList) != len(perturbList):
            raise ValueError(
                    "Length of paramList argument does not equal "
                    "length of perturbList")

        for i, (var, param, list_idx, comp_idx) in enumerate(sens_data_list):
            con = paramConst[i+1]
            if comp_idx is _NotAnIndex:
                ptb = value(perturbList[list_idx])
            else:
                try:
                    ptb = value(perturbList[list_idx][comp_idx])
                except TypeError:
                    # If the user provided a scalar value to perturb
                    # an indexed component.
                    ptb = value(perturbList[list_idx])

            # sipopt
            instance.sens_state_value_1[var] = ptb

            # k_aug
            #instance.DeltaP[con] = value(ptb - var)
            instance.DeltaP[con] = value(var - ptb)
            # FIXME: ^ This is incorrect. DeltaP should be (ptb - current).
            # But at least one test doesn't pass unless I use (current - ptb).
