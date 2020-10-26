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
from pyomo.core.expr.numvalue import (is_fixed,value)
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

@SolverFactory.register('mosek_persistent', doc = 'Persistent python interface to MOSEK.')
class MOSEKPersistent(PersistentSolver, MOSEKDirect):
    
    def __init__(self, **kwds):
        kwds['type'] = 'mosek_persistent'
        MOSEKDirect.__init__(self, **kwds)

        self._pyomo_model = kwds.pop('model', None)
        if self._pyomo_model is not None:
            self.set_instance(self._pyomo_model, **kwds)

    def _warm_start(self):
        MOSEKDirect._warm_start(self)
    
    def remove_var(self, solver_var):
        self.remove_vars(solver_var)
    
    def remove_vars(self, *solver_vars):
        try:
            var_ids = [self._pyomo_var_to_solver_var_map[v] 
                        for v in solver_vars] 
        except KeyError as ev:
            v_name = self._symbol_map.getSymbol(ev,self._labeler)
            raise ValueError(
            "Variable {} needs to be added before removal.".format(v_name))
        self._solver_model.removevars(var_ids)
        for v in solver_vars:
            self._symbol_map.removeSymbol(v)
            self._labeler.remove_obj(v)
            del self._referenced_variables[v]
            del self._pyomo_var_to_solver_var_map[v]
        var_num = self._solver_model.getnumvar()
        self._solver_var_to_pyomo_var_map = dict(zip((
            range(var_num)),self._pyomo_var_to_solver_var_map.keys()))
        self._vars_append_offset = var_num

    def remove_constraint(self, solver_con):
        self.remove_constraints(solver_con)
    
    def remove_constraints(self, *solver_cons):
        """
        Method to remove a constraint/cone from the solver model.
        User can pass several constraints as arguments, each of them 
        separated by commas (for instance, unpack a list of constraints).
        It is important to remember that removing constraints is an 
        expensive operation, that may not necessarily provide a huge 
        advantage over re-doing the entire model from scratch, especially
        if the Interior-point optimizer is being used.
        """
        lq = list(itertools.filterfalse(
            lambda x: isinstance(x,_ConicBase), solver_cons))
        cones = list(filter(lambda x: isinstance(x,_ConicBase), solver_cons))
        try:
            lq = [self._pyomo_con_to_solver_con_map[c] for c in lq]
            cones = [self._pyomo_cone_to_solver_cone_map[c] for c in cones]
        except KeyError as ec:
            c_name = self._symbol_map.getSymbol(ec, self._labeler)
            raise ValueError(
            "Constraint/Cone {} needs to be added before removal.".format(c_name))
        self._solver_model.removecons(lq)
        self._solver_model.removecones(cones)
        lq_num = self._solver_model.getnumcon()
        cone_num = self._solver_model.getnumcone()
        for c in lq:
            self._symbol_map.removeSymbol(c)
            self._labeler.remove_obj(c)
            del self._pyomo_con_to_solver_con_map[c]
        for c in cones:
            self._symbol_map.removeSymbol(c)
            self._labeler.remove_obj(c)
            del self._pyomo_cone_to_solver_cone_map[c]
        self._solver_con_to_pyomo_con_map = dict(zip(
            range(lq_num),self._pyomo_con_to_solver_con_map.keys()))
        self._solver_cone_to_pyomo_cone_map = dict(zip(
            range(cone_num),self._pyomo_cone_to_solver_cone_map.keys()))
        self._cons_append_offset = lq_num
        self._cones_append_offset = cone_num
        for c in solver_cons:
            for v in self._vars_referenced_by_con[c]:
                self._referenced_variables[v] -= 1
    
    def update_var(self, solver_var):
        self.update_vars(solver_var)
        
    def update_vars(self, *solver_vars):
        """
        Update variable(s) in the solver's model.
        This method allows fixing/unfixing variables, changing the 
        variable bounds and updating the variable types.
        Passing a single scalar var, much like other interfaces
        is perfectly valid, but a user can also pass several variables
        as arguments to change several variables at a time. This functionality
        has been introduced with the list unpacking operator in mind.

        Parameters:
        *solver_vars: scalar Vars or single _VarData, separated by commas
        """
        try:
            var_ids = [self._pyomo_var_to_solver_var_map[v] for v in solver_vars]
        except KeyError as ev:
            v_name = self._symbol_map.getSymbol(ev, self._labeler)
            raise ValueError(
            "Variable {} needs to be added before it can be modified.".format(
                v_name))
        vtypes = list(map(self._mosek_vartype_from_var, solver_vars))
        lbs, ubs, bound_types = zip(*[self._mosek_bounds(
            *p.bounds) for p in solver_vars])
        self._solver_model.putvartypelist(var_ids, vtypes)
        self._solver_model.putvarboundlist(var_ids, bound_types, lbs, ubs)
    
    def _add_column(self, var, obj_coef, constraints, coefficients):
        self._add_var(var)
        self._solver_model.putcj(self._vars_append_offset-1, obj_coef)
        self._solver_model.putacol(
            self._vars_append_offset-1, constraints, coefficients)
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