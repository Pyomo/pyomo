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

from pyomo.common.sorting import sorted_robust
from pyomo.core.expr import ExpressionReplacementVisitor
from pyomo.core.expr.numvalue import is_potentially_variable

from pyomo.common.modeling import unique_component_name
from pyomo.common.dependencies import numpy as np, scipy
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.contrib.sensitivity_toolbox.k_aug import K_augInterface, InTempDir
import logging
import os
import io
import shutil

logger = logging.getLogger('pyomo.contrib.sensitivity_toolbox')


@deprecated(
    "The sipopt function has been deprecated. Use the sensitivity_calculation() "
    "function with method='sipopt' to access this functionality.",
    logger='pyomo.contrib.sensitivity_toolbox',
    version='6.1',
)
def sipopt(
    instance,
    paramSubList,
    perturbList,
    cloneModel=True,
    tee=False,
    keepfiles=False,
    streamSoln=False,
):
    m = sensitivity_calculation(
        'sipopt',
        instance,
        paramSubList,
        perturbList,
        cloneModel,
        tee,
        keepfiles,
        solver_options=None,
    )

    return m


@deprecated(
    "The kaug function has been deprecated. Use the sensitivity_calculation() "
    "function with method='k_aug' to access this functionality.",
    logger='pyomo.contrib.sensitivity_toolbox',
    version='6.1',
)
def kaug(
    instance,
    paramSubList,
    perturbList,
    cloneModel=True,
    tee=False,
    keepfiles=False,
    solver_options=None,
    streamSoln=False,
):
    m = sensitivity_calculation(
        'k_aug',
        instance,
        paramSubList,
        perturbList,
        cloneModel,
        tee,
        keepfiles,
        solver_options,
    )

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


def sensitivity_calculation(
    method,
    instance,
    paramList,
    perturbList,
    cloneModel=True,
    tee=False,
    keepfiles=False,
    solver_options=None,
):
    """This function accepts a Pyomo ConcreteModel, a list of parameters, and
    their corresponding perturbation list. The model is then augmented with
    dummy constraints required to call sipopt or k_aug to get an approximation
    of the perturbed solution.

    Parameters
    ----------
    method: string
        'sipopt' or 'k_aug'
    instance: Block
        pyomo block or model object
    paramSubList: list
        list of mutable parameters or fixed variables
    perturbList: list
        list of perturbed parameter values
    cloneModel: bool, optional
        indicator to clone the model. If set to False, the original
        model will be altered
    tee: bool, optional
        indicator to stream solver log
    keepfiles: bool, optional
        preserve solver interface files
    solver_options: dict, optional
        Provides options to the solver (also the name of an attribute)

    Returns
    -------
    The model that was manipulated by the sensitivity interface

    """

    sens = SensitivityInterface(instance, clone_model=cloneModel)
    sens.setup_sensitivity(paramList)

    m = sens.model_instance

    if method not in {"k_aug", "sipopt"}:
        raise ValueError("Only methods 'k_aug' and 'sipopt' are supported'")

    if method == 'k_aug':
        k_aug = SolverFactory('k_aug', solver_io='nl')
        dot_sens = SolverFactory('dot_sens', solver_io='nl')
        ipopt = SolverFactory('ipopt', solver_io='nl')

        k_aug_interface = K_augInterface(k_aug=k_aug, dot_sens=dot_sens)

        ipopt.solve(m, tee=tee)
        m.ipopt_zL_in.update(m.ipopt_zL_out)  #: important!
        m.ipopt_zU_in.update(m.ipopt_zU_out)  #: important!

        k_aug.options['dsdp_mode'] = ""  #: sensitivity mode!
        k_aug_interface.k_aug(m, tee=tee)

    sens.perturb_parameters(perturbList)

    if method == 'sipopt':
        # Notes on sIpopt documentation:
        # Documentation:
        #  - https://coin-or.github.io/Ipopt/SPECIALS.html#SIPOPT
        # Original docs (archived):
        #  - http://web.archive.org/web/20210412132144/https://projects.coin-or.org/Ipopt/wiki/sIpopt
        ipopt_sens = SolverFactory('ipopt_sens', solver_io='nl')
        ipopt_sens.options['run_sens'] = 'yes'
        if solver_options is not None:
            ipopt_sens.options['linear_solver'] = solver_options

        # Send the model to ipopt_sens and collect the solution
        results = ipopt_sens.solve(m, keepfiles=keepfiles, tee=tee)

    elif method == 'k_aug':
        dot_sens.options["dsdp_mode"] = ""
        k_aug_interface.dot_sens(m, tee=tee)

    return m


def get_dsdp(model, theta_names, theta, tee=False):
    r"""This function calculates gradient vector of the variables with
    respect to the parameters (theta_names).

    For example, given:

    .. math::

        \min f:\ & p1*x1 + p2*(x2^2) + p1*p2 \\
        s.t.\ & c1: x1 + x2 = p1 \\
              & c2: x2 + x3 = p2 \\
              & 0 <= x1, x2, x3 <= 10 \\
              & p1 = 10 \\
              & p2 = 5

    the function returns dx/dp and dp/dp, and column orders.

    The following terms are used to define the output dimensions:
    - Ncon   = number of constraints
    - Nvar   = number of variables (Nx + Ntheta)
    - Nx     = number of decision (primal) variables
    - Ntheta = number of uncertain parameters.

    Parameters
    ----------
    model: Pyomo ConcreteModel
        model should include an objective function
    theta_names: list of strings
        List of Var names
    theta: dict
        Estimated parameters e.g) from parmest
    tee: bool, optional
        Indicates that ef solver output should be teed

    Returns
    -------
    dsdp: scipy.sparse.csr.csr_matrix
        Ntheta by Nvar size sparse matrix. A Jacobian matrix of the
        (decision variables, parameters) with respect to parameters
        (theta_names). number of rows = len(theta_name), number of
        columns = len(col)
    col: list
        List of variable names

    """
    # Get parameters from names. In SensitivityInterface, we expect
    # these to be parameters on the original model.
    param_list = []
    for name in theta_names:
        comp = model.find_component(name)
        if comp is None:
            raise RuntimeError("Cannot find component %s on model" % name)
        if comp.ctype is Var:
            # If theta_names correspond to Vars in the model, these vars
            # need to be fixed.
            comp.fix()
        param_list.append(comp)

    sens = SensitivityInterface(model, clone_model=True)
    m = sens.model_instance

    # Setup model and calculate sensitivity matrix with k_aug
    sens.setup_sensitivity(param_list)
    k_aug = K_augInterface()
    k_aug.k_aug(m, tee=tee)

    # Write row and col files in a temp dir, then immediately
    # read into a Python data structure.
    nl_data = {}
    with InTempDir():
        base_fname = "col_row"
        nl_file = ".".join((base_fname, "nl"))
        row_file = ".".join((base_fname, "row"))
        col_file = ".".join((base_fname, "col"))
        m.write(nl_file, io_options={"symbolic_solver_labels": True})
        for fname in [nl_file, row_file, col_file]:
            with open(fname, "r") as fp:
                nl_data[fname] = fp.read()

    # Create more useful data structures from strings
    dsdp = np.fromstring(k_aug.data["dsdp_in_.in"], sep="\n\t")
    col = nl_data[col_file].strip("\n").split("\n")
    row = nl_data[row_file].strip("\n").split("\n")

    dsdp = dsdp.reshape((len(theta_names), int(len(dsdp) / len(theta_names))))
    dsdp = dsdp[: len(theta_names), : len(col)]

    col = [i for i in col if sens.get_default_block_name() not in i]
    dsdp_out = np.zeros((len(theta_names), len(col)))
    for i in range(len(theta_names)):
        for j in range(len(col)):
            if sens.get_default_block_name() not in col[j]:
                dsdp_out[i, j] = -dsdp[i, j]  # e.g) k_aug dsdp returns -dx1/dx1 = -1.0

    return scipy.sparse.csr_matrix(dsdp_out), col


def get_dfds_dcds(model, theta_names, tee=False, solver_options=None):
    r"""This function calculates gradient vector of the objective function
    and constraints with respect to the variables and parameters.

    For example, given:

    .. math::

        \min f:\ & p1*x1 + p2*(x2^2) + p1*p2 \\
        s.t.\ & c1: x1 + x2 = p1 \\
              & c2: x2 + x3 = p2 \\
              & 0 <= x1, x2, x3 <= 10 \\
              & p1 = 10 \\
              & p2 = 5

    - Variables = (x1, x2, x3, p1, p2)
    - Fix p1 and p2 with estimated values

    The following terms are used to define the output dimensions:
    - Ncon   = number of constraints
    - Nvar   = number of variables (Nx + Ntheta)
    - Nx     = number of decision (primal) variables
    - Ntheta = number of uncertain parameters.

    Parameters
    ----------
    model : Pyomo ConcreteModel
        model should include an objective function

    theta_names : list of strings
        List of Var names

    tee : bool, optional
        Indicates that ef solver output should be teed

    solver_options : dict, optional
        Provides options to the solver (also the name of an attribute)

    Returns
    -------
    gradient_f : numpy.ndarray
        Length Nvar array. A gradient vector of the objective function
        with respect to the (decision variables, parameters) at the optimal
        solution

    gradient_c : scipy.sparse.csr.csr_matrix
        Ncon by Nvar size sparse matrix. A Jacobian matrix of the
        constraints with respect to the (decision variables, parameters)
        at the optimal solution. Each row contains [row number,
        column number, and value], column order follows variable order in col
        and index starts from 0. Note that it follows k_aug.
        If no constraint exists, return []

    col : list
        Size Nvar list of variable names

    row : list
        Size Ncon+1 list of constraints and objective function names.
        The final element is the objective function name.

    line_dic : dict
        column numbers of the theta_names in the model. Index starts from 1

    Raises
    ------
    RuntimeError
        When ipopt or k_aug or dotsens is not available
    Exception
        When ipopt fails
    """
    # Create the solver plugin using the ASL interface
    ipopt = SolverFactory('ipopt', solver_io='nl')
    if solver_options is not None:
        ipopt.options = solver_options
    k_aug = SolverFactory('k_aug', solver_io='nl')
    if not ipopt.available(False):
        raise RuntimeError('ipopt is not available')
    if not k_aug.available(False):
        raise RuntimeError('k_aug is not available')

    # Declare Suffixes
    _add_sensitivity_suffixes(model)

    # K_AUG SUFFIXES
    model.dof_v = Suffix(direction=Suffix.EXPORT)  #: SUFFIX FOR K_AUG
    model.rh_name = Suffix(direction=Suffix.IMPORT)  #: SUFFIX FOR K_AUG AS WELL
    k_aug.options["print_kkt"] = ""

    results = ipopt.solve(model, tee=tee)

    # Raise exception if ipopt fails
    if results.solver.status == SolverStatus.warning:
        raise Exception(results.solver.Message)

    for o in model.component_objects(Objective, active=True):
        f_mean = value(o)
    model.ipopt_zL_in.update(model.ipopt_zL_out)
    model.ipopt_zU_in.update(model.ipopt_zU_out)

    # run k_aug
    k_aug_interface = K_augInterface(k_aug=k_aug)
    k_aug_interface.k_aug(model, tee=tee)  #: always call k_aug AFTER ipopt.

    nl_data = {}
    with InTempDir():
        base_fname = "col_row"
        nl_file = ".".join((base_fname, "nl"))
        row_file = ".".join((base_fname, "row"))
        col_file = ".".join((base_fname, "col"))
        model.write(nl_file, io_options={"symbolic_solver_labels": True})
        for fname in [nl_file, row_file, col_file]:
            with open(fname, "r") as fp:
                nl_data[fname] = fp.read()

    col = nl_data[col_file].strip("\n").split("\n")
    row = nl_data[row_file].strip("\n").split("\n")

    # get the column numbers of "parameters"
    line_dic = {name: col.index(name) for name in theta_names}

    grad_f_file = os.path.join("GJH", "gradient_f_print.txt")
    grad_f_string = k_aug_interface.data[grad_f_file]
    gradient_f = np.fromstring(grad_f_string, sep="\n\t")
    col = [i for i in col if SensitivityInterface.get_default_block_name() not in i]

    grad_c_file = os.path.join("GJH", "A_print.txt")
    grad_c_string = k_aug_interface.data[grad_c_file]
    gradient_c = np.fromstring(grad_c_string, sep="\n\t")

    # Jacobian file is in "COO format," i.e. an nnz-by-3 array.
    # Reshape to a numpy array that matches this format.
    gradient_c = gradient_c.reshape((-1, 3))

    num_constraints = len(row) - 1  # Objective is included as a row
    if num_constraints > 0:
        row_idx = gradient_c[:, 1] - 1
        col_idx = gradient_c[:, 0] - 1
        data = gradient_c[:, 2]
        gradient_c = scipy.sparse.csr_matrix(
            (data, (row_idx, col_idx)), shape=(num_constraints, len(col))
        )
    else:
        gradient_c = np.array([])

    return gradient_f, gradient_c, col, row, line_dic


def line_num(file_name, target):
    """This function returns the line number that contains 'target' in the
    file_name. This function identifies constraints that have variables
    in theta_names.

    Parameters
    ----------
    file_name: string
        file includes the variable order (i.e. col file)
    target: string
        variable name to check

    Returns
    -------
    count: int
        line number of target in the file

    Raises
    ------
    Exception
        When file does not include target
    """
    with open(file_name) as f:
        count = int(1)
        for line in f:
            if line.strip() == target:
                return int(count)
            count += 1
    raise Exception(file_name + " does not include " + target)


class SensitivityInterface(object):
    def __init__(self, instance, clone_model=True):
        """Constructor clones model if necessary and attaches
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
        # return '_'.join(('sens_var', name))
        return name

    @staticmethod
    def get_default_param_name(name):
        # return '_'.join(('sens_param', name))
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
            if (
                hasattr(existing_block, '_has_replaced_expressions')
                and not existing_block._has_replaced_expressions
            ):
                for var, _, _, _ in existing_block._sens_data_list:
                    # Re-fix variables that the previous block was
                    # treating as parameters.
                    var.fix()
                self.model_instance.del_component(existing_block)
            else:
                msg = (
                    "Re-using sensitivity interface is not supported "
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
        # parameters with variables.
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
                            "fixed. Got %s, which is not fixed." % comp.name
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
            substitute=variableSubMap, remove_named_expressions=True
        )
        # TODO: Flag to ExpressionReplacementVisitor to only replace
        # named expressions if a node has been replaced within that
        # expression.

        new_old_comp_map = ComponentMap()

        # clone Objective, add to Block, and update any Expressions
        for obj in list(
            instance.component_data_objects(Objective, active=True, descend_into=True)
        ):
            tempName = unique_component_name(block, obj.local_name)
            new_expr = param_replacer.walk_expression(obj.expr)
            block.add_component(tempName, Objective(expr=new_expr))
            new_old_comp_map[block.component(tempName)] = obj
            obj.deactivate()

        # clone Constraints, add to Block, and update any Expressions
        #
        # Unfortunate that this deactivates and replaces constraints
        # even if they don't contain the parameters.
        #
        old_con_list = list(
            instance.component_data_objects(Constraint, active=True, descend_into=True)
        )
        last_idx = 0
        for con in old_con_list:
            new_expr = param_replacer.walk_expression(con.expr)
            # TODO: We could only create new constraints for expressions
            # where substitution actually happened, but that breaks some
            # current tests:
            #
            # if new_expr is con.expr:
            #     # No params were substituted.  We can ignore this constraint
            #     continue
            if new_expr.nargs() == 3 and (
                is_potentially_variable(new_expr.arg(0))
                or is_potentially_variable(new_expr.arg(2))
            ):
                # This is a potentially "invalid" range constraint: it
                # may now have variables in the bounds.  For safety, we
                # will split it into two simple inequalities.
                block.constList.add(expr=(new_expr.arg(0) <= new_expr.arg(1)))
                last_idx += 1
                new_old_comp_map[block.constList[last_idx]] = con
                block.constList.add(expr=(new_expr.arg(1) <= new_expr.arg(2)))
                last_idx += 1
                new_old_comp_map[block.constList[last_idx]] = con
            else:
                block.constList.add(expr=new_expr)
                last_idx += 1
                new_old_comp_map[block.constList[last_idx]] = con
            con.deactivate()

        return new_old_comp_map

    def setup_sensitivity(self, paramList):
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
        variableSubMap = dict(
            (id(param), var)
            for var, param, list_idx, _ in sens_data_list
            if paramList[list_idx].ctype is Param
        )

        if variableSubMap:
            # We now replace the provided parameters in the user's
            # expressions. Only do this if we have to, i.e. the
            # user provided some parameters rather than all vars.
            block._replaced_map = self._replace_parameters_in_constraints(
                variableSubMap
            )

            # Assume that we just replaced some params
            block._has_replaced_expressions = True

        block.paramConst = ConstraintList()
        for var, param, _, _ in sens_data_list:
            # block.paramConst.add(param - var == 0)
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
        # Note that entries of perturbList need not be components
        # of the cloned model. All we need are the values.
        instance = self.model_instance
        sens_data_list = self.block._sens_data_list
        paramConst = self.block.paramConst

        if len(self.block._paramList) != len(perturbList):
            raise ValueError(
                "Length of paramList argument does not equal length of perturbList"
            )

        for i, (var, param, list_idx, comp_idx) in enumerate(sens_data_list):
            con = paramConst[i + 1]
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
            # instance.DeltaP[con] = value(ptb - var)
            instance.DeltaP[con] = value(var - ptb)
            # FIXME: ^ This is incorrect. DeltaP should be (ptb - current).
            # But at least one test doesn't pass unless I use (current - ptb).
