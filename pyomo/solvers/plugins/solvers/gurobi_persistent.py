#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.numvalue import value, is_fixed
from pyomo.opt.base import SolverFactory
import collections


@SolverFactory.register('gurobi_persistent', doc='Persistent python interface to Gurobi')
class GurobiPersistent(PersistentSolver, GurobiDirect):
    """
    A class that provides a persistent interface to Gurobi. Direct solver interfaces do not use any file io.
    Rather, they interface directly with the python bindings for the specific solver. Persistent solver interfaces
    are similar except that they "remember" their model. Thus, persistent solver interfaces allow incremental changes
    to the solver model (e.g., the gurobi python model or the cplex python model). Note that users are responsible
    for notifying the persistent solver interfaces when changes are made to the corresponding pyomo model.

    Keyword Arguments
    -----------------
    model: ConcreteModel
        Passing a model to the constructor is equivalent to calling the set_instance mehtod.
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
        kwds['type'] = 'gurobi_persistent'
        GurobiDirect.__init__(self, **kwds)

        self._pyomo_model = kwds.pop('model', None)
        if self._pyomo_model is not None:
            self.set_instance(self._pyomo_model, **kwds)

    def _remove_constraint(self, solver_con):
        if isinstance(solver_con, self._gurobipy.Constr):
            if self._solver_model.getAttr('NumConstrs') == 0:
                self._update()
            else:
                name = self._symbol_map.getSymbol(self._solver_con_to_pyomo_con_map[solver_con])
                if self._solver_model.getConstrByName(name) is None:
                    self._update()
        elif isinstance(solver_con, self._gurobipy.QConstr):
            if self._solver_model.getAttr('NumQConstrs') == 0:
                self._update()
            else:
                try:
                    qc_row = self._solver_model.getQCRow(solver_con)
                except self._gurobipy.GurobiError:
                    self._update()
        elif isinstance(solver_con, self._gurobipy.SOS):
            if self._solver_model.getAttr('NumSOS') == 0:
                self._update()
            else:
                try:
                    sos = self._solver_model.getSOS(solver_con)
                except self._gurobipy.GurobiError:
                    self._update()
        else:
            raise ValueError('Unrecognized type for gurobi constraint: {0}'.format(type(solver_con)))
        self._solver_model.remove(solver_con)
        self._needs_updated = True

    def _remove_sos_constraint(self, solver_sos_con):
        self._remove_constraint(solver_sos_con)
        self._needs_updated = True

    def _remove_var(self, solver_var):
        if self._solver_model.getAttr('NumVars') == 0:
            self._update()
        else:
            name = self._symbol_map.getSymbol(self._solver_var_to_pyomo_var_map[solver_var])
            if self._solver_model.getVarByName(name) is None:
                self._update()
        self._solver_model.remove(solver_var)
        self._needs_updated = True

    def _warm_start(self):
        GurobiDirect._warm_start(self)

    def update_var(self, var):
        """Update a single variable in the solver's model.

        This will update bounds, fix/unfix the variable as needed, and
        update the variable type.

        Parameters
        ----------
        var: Var (scalar Var or single _VarData)

        """
        # see PR #366 for discussion about handling indexed
        # objects and keeping compatibility with the
        # pyomo.kernel objects
        #if var.is_indexed():
        #    for child_var in var.values():
        #        self.update_var(child_var)
        #    return
        if var not in self._pyomo_var_to_solver_var_map:
            raise ValueError('The Var provided to update_var needs to be added first: {0}'.format(var))
        gurobipy_var = self._pyomo_var_to_solver_var_map[var]
        vtype = self._gurobi_vtype_from_var(var)
        lb, ub = self._gurobi_lb_ub_from_var(var)

        gurobipy_var.setAttr('lb', lb)
        gurobipy_var.setAttr('ub', ub)
        gurobipy_var.setAttr('vtype', vtype)
        self._needs_updated = True

    def write(self, filename):
        """
        Write the model to a file (e.g., and lp file).

        Parameters
        ----------
        filename: str
            Name of the file to which the model should be written.
        """
        self._solver_model.write(filename)
        self._needs_updated = False

    def update(self):
        self._update()

    def set_linear_constraint_attr(self, con, attr, val):
        """
        Set the value of an attribute on a gurobi linear constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be modified.
        attr: str
            The attribute to be modified. Options are:

                CBasis
                DStart
                Lazy

        val: any
            See gurobi documentation for acceptable values.
        """
        if attr in {'Sense', 'RHS', 'ConstrName'}:
            raise ValueError('Linear constraint attr {0} cannot be set with' +
                             ' the set_linear_constraint_attr method. Please use' +
                             ' the remove_constraint and add_constraint methods.'.format(attr))
        if self._version_major < 7:
            if (self._solver_model.getAttr('NumConstrs') == 0 or
                    self._solver_model.getConstrByName(self._symbol_map.getSymbol(con)) is None):
                self._solver_model.update()
        self._pyomo_con_to_solver_con_map[con].setAttr(attr, val)
        self._needs_updated = True

    def set_var_attr(self, var, attr, val):
        """
        Set the value of an attribute on a gurobi variable.

        Parameters
        ----------
        con: pyomo.core.base.var._GeneralVarData
            The pyomo var for which the corresponding gurobi var attribute
            should be modified.
        attr: str
            The attribute to be modified. Options are:

                Start
                VarHintVal
                VarHintPri
                BranchPriority
                VBasis
                PStart

        val: any
            See gurobi documentation for acceptable values.
        """
        if attr in {'LB', 'UB', 'VType', 'VarName'}:
            raise ValueError('Var attr {0} cannot be set with' +
                             ' the set_var_attr method. Please use' +
                             ' the update_var method.'.format(attr))
        if attr == 'Obj':
            raise ValueError('Var attr Obj cannot be set with' +
                             ' the set_var_attr method. Please use' +
                             ' the set_objective method.')
        if self._version_major < 7:
            if (self._solver_model.getAttr('NumVars') == 0 or
                    self._solver_model.getVarByName(self._symbol_map.getSymbol(var)) is None):
                self._solver_model.update()
        self._pyomo_var_to_solver_var_map[var].setAttr(attr, val)
        self._needs_updated = True

    def get_model_attr(self, attr):
        """Get the value of an attribute on the Gurobi model.

        Parameters
        ----------
        attr: str
            The attribute to get. See Gurobi documentation for
            descriptions of the attributes.

            Options are:

                NumVars
                NumConstrs
                NumSOS
                NumQConstrs
                NumgGenConstrs
                NumNZs
                DNumNZs
                NumQNZs
                NumQCNZs
                NumIntVars
                NumBinVars
                NumPWLObjVars
                ModelName
                ModelSense
                ObjCon
                ObjVal
                ObjBound
                ObjBoundC
                PoolObjBound
                PoolObjVal
                MIPGap
                Runtime
                Status
                SolCount
                IterCount
                BarIterCount
                NodeCount
                IsMIP
                IsQP
                IsQCP
                IsMultiObj
                IISMinimal
                MaxCoeff
                MinCoeff
                MaxBound
                MinBound
                MaxObjCoeff
                MinObjCoeff
                MaxRHS
                MinRHS
                MaxQCCoeff
                MinQCCoeff
                MaxQCLCoeff
                MinQCLCoeff
                MaxQCRHS
                MinQCRHS
                MaxQObjCoeff
                MinQObjCoeff
                Kappa
                KappaExact
                FarkasProof
                TuneResultCount
                LicenseExpiration
                BoundVio
                BoundSVio
                BoundVioIndex
                BoundSVioIndex
                BoundVioSum
                BoundSVioSum
                ConstrVio
                ConstrSVio
                ConstrVioIndex
                ConstrSVioIndex
                ConstrVioSum
                ConstrSVioSum
                ConstrResidual
                ConstrSResidual
                ConstrResidualIndex
                ConstrSResidualIndex
                ConstrResidualSum
                ConstrSResidualSum
                DualVio
                DualSVio
                DualVioIndex
                DualSVioIndex
                DualVioSum
                DualSVioSum
                DualResidual
                DualSResidual
                DualResidualIndex
                DualSResidualIndex
                DualResidualSum
                DualSResidualSum
                ComplVio
                ComplVioIndex
                ComplVioSum
                IntVio
                IntVioIndex
                IntVioSum

        """
        if self._needs_updated:
            self._update()
        return self._solver_model.getAttr(attr)

    def get_var_attr(self, var, attr):
        """
        Get the value of an attribute on a gurobi var.

        Parameters
        ----------
        var: pyomo.core.base.var._GeneralVarData
            The pyomo var for which the corresponding gurobi var attribute
            should be retrieved.
        attr: str
            The attribute to get. Options are:

                LB
                UB
                Obj
                VType
                VarName
                X
                Xn
                RC
                BarX
                Start
                VarHintVal
                VarHintPri
                BranchPriority
                VBasis
                PStart
                IISLB
                IISUB
                PWLObjCvx
                SAObjLow
                SAObjUp
                SALBLow
                SALBUp
                SAUBLow
                SAUBUp
                UnbdRay
        """
        if self._needs_updated:
            self._update()
        return self._pyomo_var_to_solver_var_map[var].getAttr(attr)

    def get_linear_constraint_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi linear constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. Options are:

                Sense
                RHS
                ConstrName
                Pi
                Slack
                CBasis
                DStart
                Lazy
                IISConstr
                SARHSLow
                SARHSUp
                FarkasDual
        """
        if self._needs_updated:
            self._update()
        return self._pyomo_con_to_solver_con_map[con].getAttr(attr)

    def get_sos_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi sos constraint.

        Parameters
        ----------
        con: pyomo.core.base.sos._SOSConstraintData
            The pyomo SOS constraint for which the corresponding gurobi SOS constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. Options are:

                IISSOS
        """
        if self._needs_updated:
            self._update()
        return self._pyomo_con_to_solver_con_map[con].getAttr(attr)

    def get_quadratic_constraint_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi quadratic constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. Options are:

                QCSense
                QCRHS
                QCName
                QCPi
                QCSlack
                IISQConstr
        """
        if self._needs_updated:
            self._update()
        return self._pyomo_con_to_solver_con_map[con].getAttr(attr)

    def set_gurobi_param(self, param, val):
        """
        Set a gurobi parameter.

        Parameters
        ----------
        param: str
            The gurobi parameter to set. Options include any gurobi parameter.
            Please see the Gurobi documentation for options.
        val: any
            The value to set the parameter to. See Gurobi documentation for possible values.
        """
        self._solver_model.setParam(param, val)

    def get_gurobi_param_info(self, param):
        """
        Get information about a gurobi parameter.

        Parameters
        ----------
        param: str
            The gurobi parameter to get info for. See Gurobi documenation for possible options.

        Returns
        -------
        six-tuple containing the parameter name, type, value, minimum value, maximum value, and default value.
        """
        return self._solver_model.getParamInfo(param)

    def _intermediate_callback(self):
        def f(gurobi_model, where):
            self._callback_func(self._pyomo_model, self, where)
        return f

    def set_callback(self, func=None):
        r"""Specify a callback for gurobi to use.

        Parameters
        ----------
        func: function
            The function to call. The function should have three
            arguments. The first will be the pyomo model being
            solved. The second will be the GurobiPersistent
            instance. The third will be an enum member of
            gurobipy.GRB.Callback. This will indicate where in the
            branch and bound algorithm gurobi is at. For example,
            suppose we want to solve

            .. math::
               :nowrap:

               \begin{array}{ll}
               \min          & 2x + y           \\
               \mathrm{s.t.} & y \geq (x-2)^2   \\
                             & 0 \leq x \leq 4  \\
                             & y \geq 0         \\
                             & y \in \mathbb{Z}
               \end{array}

            as an MILP using exteneded cutting planes in callbacks.

            .. testcode::
               :skipif: not gurobipy_available

               from gurobipy import GRB
               import pyomo.environ as pe
               from pyomo.core.expr.taylor_series import taylor_series_expansion

               m = pe.ConcreteModel()
               m.x = pe.Var(bounds=(0, 4))
               m.y = pe.Var(within=pe.Integers, bounds=(0, None))
               m.obj = pe.Objective(expr=2*m.x + m.y)
               m.cons = pe.ConstraintList()  # for the cutting planes

               def _add_cut(xval):
                   # a function to generate the cut
                   m.x.value = xval
                   return m.cons.add(m.y >= taylor_series_expansion((m.x - 2)**2))

               _add_cut(0)  # start with 2 cuts at the bounds of x
               _add_cut(4)  # this is an arbitrary choice

               opt = pe.SolverFactory('gurobi_persistent')
               opt.set_instance(m)
               opt.set_gurobi_param('PreCrush', 1)
               opt.set_gurobi_param('LazyConstraints', 1)

               def my_callback(cb_m, cb_opt, cb_where):
                   if cb_where == GRB.Callback.MIPSOL:
                       cb_opt.cbGetSolution(vars=[m.x, m.y])
                       if m.y.value < (m.x.value - 2)**2 - 1e-6:
                           cb_opt.cbLazy(_add_cut(m.x.value))

               opt.set_callback(my_callback)
               opt.solve()

            .. testoutput::
               :hide:

               ...

            .. doctest::
               :skipif: not gurobipy_available

               >>> assert abs(m.x.value - 1) <= 1e-6
               >>> assert abs(m.y.value - 1) <= 1e-6
        """
        if func is not None:
            self._callback_func = func
            self._callback = self._intermediate_callback()
        else:
            self._callback = None
            self._callback_func = None

    def cbCut(self, con):
        """
        Add a cut within a callback.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The cut to add
        """
        if not con.active:
            raise ValueError('cbCut expected an active constraint.')

        if is_fixed(con.body):
            raise ValueError('cbCut expected a non-trival constraint')

        gurobi_expr, referenced_vars = self._get_expr_from_pyomo_expr(con.body, self._max_constraint_degree)

        if con.has_lb():
            if con.has_ub():
                raise ValueError('Range constraints are not supported in cbCut.')
            if not is_fixed(con.lower):
                raise ValueError('Lower bound of constraint {0} is not constant.'.format(con))
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError('Upper bound of constraint {0} is not constant.'.format(con))

        if con.equality:
            self._solver_model.cbCut(lhs=gurobi_expr, sense=self._gurobipy.GRB.EQUAL,
                                     rhs=value(con.lower))
        elif con.has_lb() and (value(con.lower) > -float('inf')):
            self._solver_model.cbCut(lhs=gurobi_expr, sense=self._gurobipy.GRB.GREATER_EQUAL,
                                     rhs=value(con.lower))
        elif con.has_ub() and (value(con.upper) < float('inf')):
            self._solver_model.cbCut(lhs=gurobi_expr, sense=self._gurobipy.GRB.LESS_EQUAL,
                                     rhs=value(con.upper))
        else:
            raise ValueError('Constraint does not have a lower or an upper bound {0} \n'.format(con))

    def cbGet(self, what):
        return self._solver_model.cbGet(what)

    def cbGetNodeRel(self, vars):
        """
        Parameters
        ----------
        vars: Var or iterable of Var
        """
        if not isinstance(vars, collections.Iterable):
            vars = [vars]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[i] for i in vars]
        var_values = self._solver_model.cbGetNodeRel(gurobi_vars)
        for i, v in enumerate(vars):
            v.value = var_values[i]

    def cbGetSolution(self, vars):
        """
        Parameters
        ----------
        vars: iterable of vars
        """
        if not isinstance(vars, collections.Iterable):
            vars = [vars]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[i] for i in vars]
        var_values = self._solver_model.cbGetSolution(gurobi_vars)
        for i, v in enumerate(vars):
            v.value = var_values[i]

    def cbLazy(self, con):
        """
        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The lazy constraint to add
        """
        if not con.active:
            raise ValueError('cbLazy expected an active constraint.')

        if is_fixed(con.body):
            raise ValueError('cbLazy expected a non-trival constraint')

        gurobi_expr, referenced_vars = self._get_expr_from_pyomo_expr(con.body, self._max_constraint_degree)

        if con.has_lb():
            if con.has_ub():
                raise ValueError('Range constraints are not supported in cbLazy.')
            if not is_fixed(con.lower):
                raise ValueError('Lower bound of constraint {0} is not constant.'.format(con))
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError('Upper bound of constraint {0} is not constant.'.format(con))

        if con.equality:
            self._solver_model.cbLazy(lhs=gurobi_expr, sense=self._gurobipy.GRB.EQUAL,
                                      rhs=value(con.lower))
        elif con.has_lb() and (value(con.lower) > -float('inf')):
            self._solver_model.cbLazy(lhs=gurobi_expr, sense=self._gurobipy.GRB.GREATER_EQUAL,
                                      rhs=value(con.lower))
        elif con.has_ub() and (value(con.upper) < float('inf')):
            self._solver_model.cbLazy(lhs=gurobi_expr, sense=self._gurobipy.GRB.LESS_EQUAL,
                                      rhs=value(con.upper))
        else:
            raise ValueError('Constraint does not have a lower or an upper bound {0} \n'.format(con))

    def cbSetSolution(self, vars, solution):
        if not isinstance(vars, collections.Iterable):
            vars = [vars]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[i] for i in vars]
        self._solver_model.cbSetSolution(gurobi_vars, solution)

    def cbUseSolution(self):
        return self._solver_model.cbUseSolution()

    def _add_column(self, var, obj_coef, constraints, coefficients):
        """Add a column to the solver's model

        This will add the Pyomo variable var to the solver's
        model, and put the coefficients on the associated 
        constraints in the solver model. If the obj_coef is
        not zero, it will add obj_coef*var to the objective 
        of the solver's model.

        Parameters
        ----------
        var: Var (scalar Var or single _VarData)
        obj_coef: float
        constraints: list of solver constraints
        coefficients: list of coefficients to put on var in the associated constraint
        """

        ## set-up add var
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vtype = self._gurobi_vtype_from_var(var)
        lb, ub = self._gurobi_lb_ub_from_var(var)

        gurobipy_var = self._solver_model.addVar(obj=obj_coef, lb=lb, ub=ub, vtype=vtype, name=varname, 
                            column=self._gurobipy.Column(coeffs=coefficients, constrs=constraints) )

        self._pyomo_var_to_solver_var_map[var] = gurobipy_var 
        self._solver_var_to_pyomo_var_map[gurobipy_var] = var
        self._referenced_variables[var] = len(coefficients)

    def reset(self):
        self._solver_model.reset()
