#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from __future__ import division

import logging
import tempfile
import os

from pyomo.core import *
from pyomo.dae import *
from pyomo.environ import SolverFactory

from six import iterkeys, itervalues, iteritems

logger = logging.getLogger('pyomo.core')

def initialize_by_element(model, solver, **kwds):
    """
    This function initializes a discretized DAE model by solving the DAE
    equations for each finite element or chunk of finite elements in a
    sequence.

    This is the simulation-based version of the element-by-element
    initialization strategy described by Larry. We should be solving a
    sequence of square systems. The user is responsible for providing
    initial values for all differential variables and for initializing
    AND FIXING any control variables. 

    """

    # Check if model has been fully discretized
    for t in model.component_objects(ctype=ContinuousSet):
        if len(t.get_discretization_info()) == 0:
            raise DAE_Error("The initialization function may only be "
                            "applied to fully discretized models")

    # ContinuousSet representing time
    time = kwds.pop('contset',None)
    if time is None:
        raise DAE_Error("You must identify the ContinuousSet representing time")

    if type(time) is str:
        time = model.component(time)

    if not isinstance(time, ContinuousSet):
        raise DAE_Error("Expected a ContinuousSet componenet to be passed "
                        "to the keyword argument 'contset'")

    # number of finite elements per chunk 
    nchunk = kwds.pop('groupby', 1)
    
    # Divide points in the time ContinuousSet into chunks, going to be
    # overlap with finite element boundaries. With points pre-divided we
    # can identify Vars and Constraints belonging to the chunks and
    # separate them out. 

    elements = time.get_finite_elements()
    numgroups = int(ceil((len(elements)-1)/nchunk))
    timegroups = {}

    # print('# groups: ', numgroups)
    # print(elements)

    for i in range(numgroups):
        
        lower = elements[i*nchunk]
        try:
            upper = elements[(i+1)*nchunk]
        except IndexError:
            upper = elements[-1]

        #print(lower,' <= j <= ', upper)
        points = [j for j in time if j >= lower and j<= upper]
        
        timegroups[i] = points
    print(timegroups)

    ######################
    # Initial Book-keeping
    ######################

    # Loop over all constraints
    con_state = []
    for con in model.component_data_objects(ctype=Constraint, active=True):
        # Save original state of all constraints
        con_state.append((con, con.active))
        # Deactivate inequality constraints
        if not con.equality:
            con.deactivate()

    # Find Time-indexed constraints
    #    determine order of time in indexing sets
    # The only constraints coupling the elements should be the
    # discretization equations

    conlist = []
    couplinglist = []
    for con in model.component_objects(ctype=Constraint, active=True):
        if con.dim() == 0:
            continue

        # Deactivate constraints not indexed by time?? For now just
        # ignore them
        if con._implicit_subsets is None:
            if time is con._index:
                if '_disc_eq' in con.name:
                    couplinglist.append((con,None))
                else:
                    conlist.append((con,None))
            else:
                con.deactivate()
        else:
            if time in con._implicit_subsets:
                dimsum = 0
                for s in con._implicit_subsets:
                    if s is time:
                        break
                    dimsum += s.dimen

                if '_disc_eq' in con.name:
                    couplinglist.append((con,dimsum))
                else:
                    conlist.append((con,dimsum))
            else:
                con.deactivate()

    # Anything to do for Expressions??
    # Anything to do for ConstraintLists??
    # Anything to do for hierarchical models??  
    # Anything to do for time indexed blocks??

    # Create lists of constraints belonging to each element
    congroups = {}
    for i in range(numgroups):
        temp = []
        for con,csidx in conlist:
            if csidx is None:
                noncsidx = (None,)
            else:
                noncsidx = set(idx[:csidx]+idx[csidx+1:] for idx in con.keys())
            for t in timegroups[i]:
                for nidx in noncsidx:
                    #try:
                    if csidx is None:
                        if t in con.keys():
                            if con[t].active:
                                con[t].deactivate()
                                temp.append(con[t])
                    else:
                        tempidx = nidx[:csidx]+(t,)+nidx[csidx:]
                        if tempidx in con.keys():
                            if con[tempidx].active:
                                con[tempidx].deactivate()
                                temp.append(con[tempidx])
                    # except KeyError:
                    #     continue
        
        for con,csidx in couplinglist:
            if csidx is None:
                noncsidx = (None,)
            else:
                noncsidx = set(idx[:csidx]+idx[csidx+1:] for idx in con.keys())
            # Loop over all the time point in the time group but skip
            # the end boundary. Might need to modify this to skip based
            # on discretization scheme used
            # for t in timegroups[i][:-1]: 
            for t in timegroups[i][1:]: 
                for nidx in noncsidx:
                    #try:
                    if csidx is None:
                        if t in con.keys():
                            con[t].deactivate()
                            temp.append(con[t])
                    else:
                        tempidx = nidx[:csidx]+(t,)+nidx[csidx:]
                        if tempidx in con.keys():
                            con[tempidx].deactivate()
                            temp.append(con[tempidx])
                    #except KeyError:
                    #    continue

        congroups[i] = temp

    # Loop over all variables and save original state
    var_state = []
    for var in model.component_data_objects(ctype=Var):
        var_state.append((var, var.fixed, var.value))

    # Fix variables not indexed by time to their initial values and
    # throw error if no initial value was provided

    varlist = []
    for var in model.component_objects(ctype=Var):
        
        if var.dim() == 0:
            if var.value is None:
                raise ValueError('No initial value provided for variable "%s"' %(var.name))
            var.fix()
            continue

        # Skip DerivativeVars
        if type(var) is DerivativeVar:
            continue

        if var._implicit_subsets is None:
            if time is var._index:
                # Skip variables that have been fixed
                if all(var[i].fixed for i in var):
                    continue
                varlist.append((var,None))
            else:
                var.fix()
        else:
            if time in var._implicit_subsets:
                dimsum = 0
                for s in var._implicit_subsets:
                    if s is time:
                        break
                    dimsum += s.dimen

                # Skip variables that have been fixed
                if all(var[i].fixed for i in var):
                    continue
                varlist.append((var,dimsum))
            else:
                # Should check for initial values??
                var.fix()

    # Create lists of variables to be fixed for each element
    vargroups = {}
    for i in range(numgroups):
        temp = []
        for var,csidx in varlist:
            if csidx is None:
                noncsidx = (None,)
            else:
                noncsidx = set(idx[:csidx]+idx[csidx+1:] for idx in var.keys())
            # Get the first time point in the time group
            t = min(timegroups[i])
            for nidx in noncsidx:
                try:
                    if csidx is None:
                        var[t].unfix()
                        temp.append(var[t])
                    else:
                        tempidx = nidx[:csidx]+(t,)+nidx[csidx:]
                        var[tempidx].unfix()
                        temp.append(var[tempidx])
                except KeyError:
                    continue
        
        vargroups[i] = temp

    # Deactivate the objective function(s)
    obj_list = list(model.component_objects(ctype=Objective, active=True))
    for o in obj_list:
        o.deactivate()

    sol = SolverFactory(solver)
    
    # For each finite element chunk
    for i in range(numgroups):
        print('Solving element ', i)

        # Fix the differential variable values at the beginning of the chunk
        for v in vargroups[i]:
            v.fix()

        # Activate time-indexed constraints over chunk
        for c in congroups[i]:
            c.activate()

        # Check if square
        _check_if_square(model)

        # Solve
        sol.solve(model)

        # Check if solve successful?

        # Deactivate constraints
        for c in congroups[i]:
            c.deactivate()

        # Unfix differential variables
        for v in vargroups[i]:
            v.unfix()

    # At the end, set the state of the model back to it's original state
    for con, active in con_state:
        if active:
            con.activate()
        else:
            con.deactivate()
        # con.active = active
    
    for var, fixed, value in var_state:
        var.fixed = fixed
        #var.value = value # Check this later, might not be needed

    # Reactivate the objective function(s)
    for o in obj_list:
        o.activate()


def _check_if_square(model):
    
    fid, fname = tempfile.mkstemp()
    os.close(fid)

    # This is so ugly. Why can't it just return the symbol map?!
    fname_, smap_id = model.write(filename=fname, format="nl")

    symbol_map = model.solutions.symbol_map[smap_id]
    model.solutions.delete_symbol_map(smap_id)

    # delete the temp file
    os.remove(fname)

    # used things
    used_vars = 0
    used_equality_cons = 0
    used_inequality_cons = 0
    for obj_weakref in symbol_map.bySymbol.values():
        obj = obj_weakref()
        assert obj is not None
        if obj.parent_component().type() is Var:
            used_vars += 1
        elif obj.parent_component().type() is Constraint:
            if obj.equality:
                used_equality_cons += 1
            else:
                used_inequality_cons += 1

        else:
            assert (obj.parent_component().type() is Objective) or \
                (obj.parent_component().type() is Suffix), str(type(obj))

    print('# Vars: ', used_vars)
    print('# Cons: ', used_equality_cons)

    if used_vars != used_equality_cons:
        raise DAE_Error('The current subproblem is not square. Please double-check your formulation')
