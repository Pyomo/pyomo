#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import operator
import itertools
import pyomo.core.base.var
import pyomo.core.base.constraint
from pyomo.core import is_fixed, value
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.solvers.plugins.solvers.mosek_direct import MOSEKDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import \
    DirectOrPersistentSolver
from pyomo.opt.base import SolverFactory
from pyomo.core.kernel.conic import _ConicBase
from pyomo.core.kernel.block import block


@SolverFactory.register('mosek_persistent', doc='Persistent python interface to MOSEK.')
class MOSEKPersistent(PersistentSolver, MOSEKDirect):
    """
    This class provides a persistent interface between pyomo and MOSEK's Optimizer API.
    As a child to the MOSEKDirect class, this interface does not need any file IO. 
    Furthermore, the persistent interface preserves the MOSEK task object, allowing 
    users to make incremental changes (such as removing variables/constraints, modifying
    variables, adding columns etc.) to their models. Note that users are responsible for 
    informing the persistent interface of any incremental change. For instance, if a new 
    variable is defined, then it would need to be added explicitly by calling the add_var
    method, before the solver knows of its existence.
    Keyword Arguments
    -----------------
    type: str
        String indicating the class type of the solver instance.
    name: str
        String representing either the class type of the solver instance or an assigned name.
    doc: str
        Documentation for the solver
    options: dict
        Dictionary of solver options
    """

    def __init__(self, **kwds):
        kwds.setdefault('type', 'mosek_persistent')
        MOSEKDirect.__init__(self, **kwds)

        self._pyomo_model = kwds.pop('model', None)
        if self._pyomo_model is not None:
            self.set_instance(self._pyomo_model, **kwds)

    def _warm_start(self):
        MOSEKDirect._warm_start(self)

    def add_vars(self, var_seq):
        """
        Add multiple variables to the MOSEK task object in one method call.

        This will keep any existing model components intact.

        Parameters
        ----------
        var_seq: tuple/list of Var
        """
        self._add_vars(var_seq)

    def add_constraints(self, con_seq):
        """
        Add multiple constraints to the MOSEK task object in one method call.

        This will keep any existing model components intact.

        NOTE: If this method is used to add cones, then the cones should be 
        passed as constraints. Use the add_block method for conic_domains.

        Parameters
        ----------
        con_seq: tuple/list of Constraint (scalar Constraint or single _ConstraintData)
        """
        self._add_constraints(con_seq)

    def remove_var(self, solver_var):
        """
        Remove a single variable from the model as well as the MOSEK task.
        This will keep any other model components intact.
        Parameters
        ----------
        solver_var: Var (scalar Var or single _VarData)
        """
        self.remove_vars(solver_var)

    def remove_vars(self, *solver_vars):
        """
        Remove multiple scalar variables from the model as well as the MOSEK task. The
        user can pass an unpacked list of scalar variables.
        This will keep any other model components intact.
        Parameters
        ----------
        *solver_var: Var (scalar Var or single _VarData)
        """
        try:
            var_ids = []
            for v in solver_vars:
                var_ids.append(self._pyomo_var_to_solver_var_map[v])
                self._symbol_map.removeSymbol(v)
                self._labeler.remove_obj(v)
                del self._referenced_variables[v]
                del self._pyomo_var_to_solver_var_map[v]
            self._solver_model.removevars(var_ids)
        except KeyError:
            v_name = self._symbol_map.getSymbol(v, self._labeler)
            raise ValueError(
                "Variable {} needs to be added before removal.".format(v_name))
        var_num = self._solver_model.getnumvar()
        for i, v in enumerate(self._pyomo_var_to_solver_var_map):
            self._pyomo_var_to_solver_var_map[v] = i
        self._solver_var_to_pyomo_var_map = dict(zip((
            range(var_num)), self._pyomo_var_to_solver_var_map.keys()))

    def remove_constraint(self, solver_con):
        """
        Remove a single constraint from the model as well as the MOSEK task. 

        This will keep any other model components intact.

        To remove a conic-domain, you should use the remove_block method.
        Parameters
        ----------
        solver_con: Constraint (scalar Constraint or single _ConstraintData)
        """
        self.remove_constraints(solver_con)

    def remove_constraints(self, *solver_cons):
        """
        Remove multiple constraints from the model as well as the MOSEK task in one
        method call. 

        This will keep any other model components intact.
        To remove conic-domains, use the remove_block method.

        Parameters
        ----------
        *solver_cons: Constraint (scalar Constraint or single _ConstraintData)
        """
        lq_cons = tuple(itertools.filterfalse(
            lambda x: isinstance(x, _ConicBase), solver_cons))
        cone_cons = tuple(
            filter(lambda x: isinstance(x, _ConicBase), solver_cons))
        try:
            lq = []
            cones = []
            for c in lq_cons:
                lq.append(self._pyomo_con_to_solver_con_map[c])
                self._symbol_map.removeSymbol(c)
                self._labeler.remove_obj(c)
                del self._pyomo_con_to_solver_con_map[c]
            for c in cone_cons:
                cones.append(self._pyomo_cone_to_solver_cone_map[c])
                self._symbol_map.removeSymbol(c)
                self._labeler.remove_obj(c)
                del self._pyomo_cone_to_solver_cone_map[c]
            self._solver_model.removecons(lq)
            self._solver_model.removecones(cones)
            lq_num = self._solver_model.getnumcon()
            cone_num = self._solver_model.getnumcone()
        except KeyError:
            c_name = self._symbol_map.getSymbol(c, self._labeler)
            raise ValueError(
                "Constraint/Cone {} needs to be added before removal.".format(c_name))
        self._solver_con_to_pyomo_con_map = dict(zip(
            range(lq_num), self._pyomo_con_to_solver_con_map.keys()))
        self._solver_cone_to_pyomo_cone_map = dict(zip(
            range(cone_num), self._pyomo_cone_to_solver_cone_map.keys()))
        for i, c in enumerate(self._pyomo_con_to_solver_con_map):
            self._pyomo_con_to_solver_con_map[c] = i
        for i, c in enumerate(self._pyomo_cone_to_solver_cone_map):
            self._pyomo_cone_to_solver_cone_map[c] = i

    def update_var(self, solver_var):
        """
        Update a single variable in solver model. This method allows fixing/unfixing,
        changing variable type and bounds.
        Parameters
        ----------
        solver_var: Var
        """

        self.update_vars(solver_var)

    def update_vars(self, *solver_vars):
        """
        Update multiple scalar variables in solver model. This method allows fixing/unfixing,
        changing variable types and bounds.
        Parameters
        ----------
        *solver_var: Constraint (scalar Constraint or single _ConstraintData)
        """
        try:
            var_ids = []
            for v in solver_vars:
                var_ids.append(self._pyomo_var_to_solver_var_map[v])
            vtypes = tuple(map(self._mosek_vartype_from_var, solver_vars))
            lbs = tuple(-float('inf') if value(v.lb) is None else value(v.lb)
                        for v in solver_vars)
            ubs = tuple(float('inf') if value(v.ub) is None else value(v.ub)
                        for v in solver_vars)
            fxs = tuple(v.is_fixed() for v in solver_vars)
            bound_types = tuple(map(self._mosek_bounds, lbs, ubs, fxs))
            self._solver_model.putvartypelist(var_ids, vtypes)
            self._solver_model.putvarboundlist(var_ids, bound_types, lbs, ubs)
        except KeyError:
            print(v.name)
            v_name = self._symbol_map.getSymbol(v, self._labeler)
            raise ValueError(
                "Variable {} needs to be added before it can be modified.".format(
                    v_name))

    def _add_column(self, var, obj_coef, constraints, coefficients):
        self.add_var(var)
        var_num = self._solver_model.getnumvar()
        self._solver_model.putcj(var_num-1, obj_coef)
        self._solver_model.putacol(
            var_num-1, constraints, coefficients)
        self._referenced_variables[var] = len(constraints)

    def write(self, filename):
        """
        Write the model to a file. MOSEK can write files in various
        popular formats such as: lp, mps, ptf, cbf etc.
        In addition to the file formats mentioned above, MOSEK can 
        also write files to native formats such as : opf, task and
        jtask. The task format is binary, and is the preferred format
        for sharing with the MOSEK staff in case of queries, since it saves
        the status of the problem and the solver down the smallest detail.
        Parameters:
        filename: str (Name of the output file, including the desired extension)
        """
        if 'task' in filename.split("."):
            self._solver_model.writetask(filename)
        else:
            self._solver_model.writedata(filename)
