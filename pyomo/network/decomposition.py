#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = [ 'SequentialDecomposition' ]

from pyomo.network import Port, Arc
from pyomo.core import Constraint, value, Objective, Var, ConcreteModel, \
    Binary, minimize, Expression
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.expr.current import identify_variables
from pyomo.repn import generate_standard_repn
from pyutilib.misc import Options
import networkx as nx
import numpy
import copy, logging
from six import iteritems, itervalues

logger = logging.getLogger('pyomo.network')
logger.setLevel(logging.INFO) # TODO: option to set logging level somewhere?


class SequentialDecomposition(object):
    """
    A sequential decomposition tool for Pyomo network models.

    Options, accessed via self.options:
        graph                   A networkx graph representing the model to
                                    be solved
                                    Default: None (will compute it)
        tear_set                A list of indexes representing edges to be
                                    torn. Can be set with a list of edge
                                    tuples via set_tear_set
                                    Default: None (will compute it)
        select_tear_method      Which method to use to select a tear set,
                                    either "mip" or "heuristic"
                                    Default: "mip"
        run_first_pass          Boolean indicating whether or not to run
                                    through network before running the
                                    tear stream convergence procedure
                                    Default: True
        solve_tears             Boolean indicating whether or not to run
                                    iterations to converge tear streams
                                    Default: True
        tear_method             Method to use for converging tear streams,
                                    either "Direct" or "Wegstein"
                                    Default: "Direct"
        iterLim                 Limit on the number of tear iterations
                                    Default: 40
        tol                     Tolerance at which to stop tear iterations
                                    Default: 1.0E-5
        accelMin                Min value for Wegstein acceleration factor
                                    Default: -5
        accelMax                Max value for Wegstein acceleration factor
                                    Default: 0
        tearTolType             Type of Wegstein tolerance value, either:
                                    "abs" - Absolute tolerance
                                    "rel" - Relative tolerance (to bound range)
                                    Default: "abs"
        solver                  Name of solver to use for passing values
                                    when there is more than one free var
                                    Default: "baron"
        solver_io               Solver IO keyword for the above solver
                                    Default: None
        solver_options          Keyword options to pass to solve method
                                    Default: {}
        tear_solver             Name of solver to use for select_tear_mip
                                    Default: "cplex"
        tear_solver_io          Solver IO keyword for the above solver
                                    Default: "python"
        tear_solver_options     Keyword options to pass to solve method
                                    Default: {}
    """

    def __init__(self):
        self.cache = {}
        options = self.options = Options()
        # defaults
        options["graph"] = None
        options["tear_set"] = None
        options["select_tear_method"] = "mip"
        options["run_first_pass"] = True
        options["solve_tears"] = True
        options["tear_method"] = "Direct"
        options["iterLim"] = 40
        options["tol"] = 1.0E-5
        options["accelMin"] = -5
        options["accelMax"] = 0
        options["tearTolType"] = "abs"
        options["solver"] = "baron"
        options["solver_io"] = None
        options["solver_options"] = {}
        options["tear_solver"] = "cplex"
        options["tear_solver_io"] = "python"
        options["tear_solver_options"] = {}

    def set_tear_set(self, G, tset):
        """
        Set a custom tear set to be used when running the decomposition.

        The procedure will use this custom tear set instead of finding
        its own, thus it can save some time. Additionally, this will be
        useful for knowing which edges will need guesses

        Arguments:
            G               A networkx graph corresponding to tset
            tset            A list of tuples representing edges to tear

        Returns:
            The tear set list represented by edge indexes
        """
        edge_map = self.edge_to_idx(G)
        res = []
        for edge in tset:
            res.append(edge_map[edge])
        self.options["tear_set"] = res
        return res

    def tear_set_edges(self, G, method="mip", **kwds):
        """
        Call the specified tear selection method and return a list
        of edge tuples representing the selected tear edges.

        The **kwds will be passed to the method.
        """
        if method == "mip":
            tset = self.select_tear_mip(G, **kwds)
        elif method == "heuristic":
            # tset is the first list in the first return value
            tset = self.select_tear_heuristic(G, **kwds)[0][0]
        else:
            raise ValueError("Invalid method '%s'" % (method,))

        return self.indexes_to_edges(G, tset)

    def indexes_to_edges(self, G, lst):
        """
        Converts a list of edge indexes to the edge tuples.

        Arguments:
            G               A networkx graph corresponding to lst
            lst             A list of edge indexes to convert to tuples

        Returns:
            A list of edge tuples
        """
        edge_list = self.idx_to_edge(G)
        res = []
        for ei in lst:
            res.append(edge_list[ei])
        return res

    def run(self, model, function):
        """
        Compute a Pyomo network model using sequential decomposition.

        Arguments:
            model           A Pyomo model
            function        A function to be called on each block/node
                                in the network
        """
        self.cache.clear()

        G = self.options["graph"]
        if G is None:
            G = self.create_graph(model)

        if self.options["run_first_pass"]:
            order = self.calculation_order(G)
            self.run_order(G, order, function)

        tset = self.tear_set(G)
        if not self.options["solve_tears"] or not len(tset):
            # Not solving tears, we're done
            return

        sccNodes, sccEdges, sccOrder = self.scc_collect(G)

        for lev in sccOrder:
            for sccIndex in lev:
                order = self.calculation_order(G, nodes=sccNodes[sccIndex])

                # only pass tears that are part of this SCC
                tears = []
                for ei in sccEdges[sccIndex]:
                    if ei in tset:
                        tears.append(ei)

                kwds = dict(G=G, order=order, function=function, tears=tears,
                    iterLim=self.options["iterLim"], tol=self.options["tol"])

                tear_method = self.options["tear_method"]

                if tear_method == "Direct":
                    self.solve_tear_direct(**kwds)

                elif tear_method == "Wegstein":
                    kwds["accelMin"] = self.options["accelMin"]
                    kwds["accelMax"] = self.options["accelMax"]
                    kwds["tearTolType"] = self.options["tearTolType"]
                    self.solve_tear_wegstein(**kwds)

                else:
                    raise ValueError(
                        "Invalid tear_method '%s'" % (tear_method,))

        self.cache.clear()

    def run_order(self, G, order, function):
        """
        Run computations in the order provided by calling the function.

        Arguments:
            G               A networkx graph corresponding to order
            order           The order in which to run each node in the graph
            function        The function to be called on each block/node
        """
        fixed_inputs = self.fixed_inputs()
        fixed_outputs = ComponentSet()
        tset = self.tear_set(G)
        edge_map = self.edge_to_idx(G)
        for lev in order:
            for unit in lev:
                # make sure all inputs are fixed
                for p in unit.component_data_objects(Port):
                    if not len(p.sources()):
                        continue
                    for var in p.iter_vars(expr_vars=True, fixed=False):
                        if unit not in fixed_inputs:
                            fixed_inputs[unit] = ComponentSet()
                        fixed_inputs[unit].add(var)
                        if var.value is None:
                            var.value = 0
                        var.fix()

                function(unit)

                # free the inputs that were not already fixed
                if unit in fixed_inputs:
                    for var in fixed_inputs[unit]:
                        var.free()
                    fixed_inputs[unit].clear()

                # pass the values downstream for all outlet ports
                for p in unit.component_data_objects(Port):
                    dests = p.dests()
                    if not len(dests):
                        continue
                    for var in p.iter_vars(expr_vars=True, fixed=False):
                        fixed_outputs.add(var)
                        var.fix()
                    for arc in dests:
                        # make sure the edge (index) is not in the tear set
                        edge = (arc.src.parent_block(), arc.dest.parent_block())
                        if edge_map[edge] not in tset:
                            self.pass_values(arc, fixed_inputs)
                    for var in fixed_outputs:
                        var.free()
                    fixed_outputs.clear()

    def pass_values(self, arc, fixed_inputs):
        """
        Pass the values from one unit to the next, recording only those that
        were not already fixed in the provided dict that maps blocks to sets.
        """
        eblock = arc.expanded_block
        dest = arc.destination
        dest_unit = dest.parent_block()

        if dest_unit not in fixed_inputs:
            fixed_inputs[dest_unit] = ComponentSet()

        need_to_solve = False
        for con in eblock.component_data_objects(Constraint, active=True):
            # we expect to find equality constraints with one linear variable
            if not con.equality:
                # We assume every constraint here is an equality.
                # This will only be False if the transformation changes
                # or if the user puts something unexpected on the eblock.
                raise RuntimeError(
                    "Found inequality constraint '%s'. Please do not modify "
                    "the expanded block." % con.name)
            repn = generate_standard_repn(con.body)
            if repn.is_fixed():
                # the port member's peer was already fixed
                # TODO: what if we have two fixed but different values
                # at either end of the port?
                continue
            if not (repn.is_linear() and len(repn.linear_vars) == 1):
                # TODO: need to confirm this is the right thing to do
                need_to_solve = True
                logger.warning("Constraint '%s' has more than one free "
                    "variable." % con.name)
                continue
            # fix the value of the single variable to satisfy the constraint
            val = (con.lower - repn.constant) / repn.linear_coefs[0]
            var = repn.linear_vars[0]
            fixed_inputs[dest_unit].add(var)
            var.fix(val)

        if not need_to_solve:
            return

        raise Exception("I haven't figured this out yet")

        logger.warning(
            "Need to call solver for underspecified constraints of arc '%s'.\n"
            "This probably means either the split fraction was not specified "
            "or the next port's member was a multivariable expression.\n"
            "Values may be arbitrary." % arc.name)

        from pyomo.environ import SolverFactory
        # TODO: even if this is the right thing to do, we can't just create
        # this dummy objective on the actual model. We're gonna need to create
        # a new model if we have to solve like this.
        eblock.o = Objective(expr=1)
        opt = SolverFactory(self.options["solver"],
                            solver_io=self.options["solver_io"])
        kwds = self.options["solver_options"]
        opt.solve(eblock, **kwds)

        for name, mem in dest.iter_vars(with_names=True):
            # go and fix the rest of the newly evaluated variables
            try:
                index = mem.index()
            except AttributeError:
                index = None
            obj = self.source_dest_peer(arc, name, index)
            self.fix(obj, fixed_inputs[dest_unit])

    def source_dest_peer(self, arc, name, index=None):
        """
        Return the object that is the peer to the source port's member.
        This is either the destination port's member, or the variable
        on the arc's expanded block for Extensive properties. This will
        return the appropriate index of the peer.
        """
        # check the rule on source but dest should be the same
        if arc.src._rules[name][0] is Port.Extensive:
            evar = arc.expanded_block.component(name)
            if evar is not None:
                # 1-to-1 arcs don't make evar because they're an equality
                return evar[index]
        mem = arc.dest.vars[name]
        if mem.is_indexed():
            return mem[index]
        else:
            return mem

    def fix(self, obj, fixed):
        """
        Fix a variable or every variable in an expression,
        recording the variable if it was not already fixed.
        """
        if obj.is_expression_type():
            for var in identify_variables(obj, include_fixed=False):
                fixed.add(var)
                var.fix()
        elif not obj.is_fixed():
            fixed.add(obj)
            obj.fix()

    def create_graph(self, model):
        """
        Returns a networkx representation of a Pyomo connected network.

        The nodes are units and the edges follow Pyomo Arc objects. Nodes
        that get added to the graph are determined by the parent blocks
        of the source and destination Ports of every Arc in the model.
        Edges are added for each Arc using the direction specified by
        source and destination. All Arcs in the model will be used whether
        or not they are active (since this needs to be done after expansion),
        and they all need to be directed.
        """
        G = nx.DiGraph()

        for arc in model.component_data_objects(Arc):
            if not arc.directed:
                raise ValueError("All Arcs must be directed when creating "
                                 "a graph for a model. Found undirected "
                                 "Arc: '%s'" % arc.name)
            src, dest = arc.src.parent_block(), arc.dest.parent_block()
            if G.has_edge(src, dest):
                # multilpe arcs between two units
                raise Exception("I haven't figured this out yet")
            G.add_edge(src, dest, arc=arc)

        return G

    def select_tear_mip(self, G, solver, solver_io=None, solver_options={}):
        """
        This finds optimal sets of tear edges based on two criteria.
        The primary objective is to minimize the maximum number of
        times any cycle is broken. The seconday criteria is to
        minimize the number of tears. This function creates a MIP
        problem in Pyomo with a doubly weighted objective and solves
        it with the solver specifications in the options container.
        """
        model = ConcreteModel()

        bin_list = []
        for i in range(G.number_of_edges()):
            # add a binary "torn" variable for every edge
            vname = "edge%s" % i
            var = Var(domain=Binary)
            bin_list.append(var)
            model.add_component(vname, var)

        # var containing the maximum number of times any cycle is torn
        mct = model.max_cycle_tears = Var()

        _, cycleEdges = self.all_cycles(G)

        for i in range(len(cycleEdges)):
            ecyc = cycleEdges[i]

            # expression containing sum of tears for each cycle
            ename = "cycle_sum%s" % i
            expr = Expression(expr=sum(bin_list[i] for i in ecyc))
            model.add_component(ename, expr)

            # every cycle must have at least 1 tear
            cname_min = "cycle_min%s" % i
            con_min = Constraint(expr=expr >= 1)
            model.add_component(cname_min, con_min)

            # mct >= cycle_sum for all cycles, thus it becomes the max
            cname_mct = mct.name + "_geq%s" % i
            con_mct = Constraint(expr=mct >= expr)
            model.add_component(cname_mct, con_mct)

        # weigh the primary objective much greater than the secondary
        obj_expr = 1000 * mct + sum(var for var in bin_list)
        model.obj = Objective(expr=obj_expr, sense=minimize)

        from pyomo.environ import SolverFactory
        opt = SolverFactory(solver, solver_io=solver_io)
        opt.solve(model, **solver_options)

        # collect final list by adding every edge with a "True" binary var
        tset = []
        for i in range(len(bin_list)):
            if bin_list[i].value == 1:
                tset.append(i)

        return tset

    def tear_error(self, G, tears):
        """
        Returns a numpy array of differences between src and dest values
        for all edges in the tears list of edge indexes.
        """
        svals = []
        dvals = []
        edge_list = self.idx_to_edge(G)
        for tear in tears:
            arc = G.edges[edge_list[tear]]["arc"]
            src, dest = arc.src, arc.dest
            sf = arc.expanded_block.component("splitfrac")
            for name, mem in src.iter_vars(with_names=True):
                if sf is not None:
                    svals.append(value(mem * sf))
                else:
                    svals.append(value(mem))
                try:
                    index = mem.index()
                except AttributeError:
                    index = None
                dvals.append(value(self.source_dest_peer(arc, name, index)))
        svals = numpy.array(svals)
        dvals = numpy.array(dvals)
        return svals - dvals

    def pass_tear_weg(self, G, tears, x):
        """
        Set the destination value of all tear edges to the corresponding
        value in the numpy array x.
        """
        fixed_inputs = self.fixed_inputs()
        edge_list = self.idx_to_edge(G)
        i = 0
        for tear in tears:
            arc = G.edges[edge_list[tear]]["arc"]
            src, dest = arc.src, arc.dest
            dest_unit = dest.parent_block()

            if dest_unit not in fixed_inputs:
                fixed_inputs[dest_unit] = ComponentSet()

            need_to_solve = False
            for name, mem in src.iter_vars(with_names=True):
                try:
                    index = mem.index()
                except AttributeError:
                    index = None
                obj = self.source_dest_peer(arc, name, index)
                if obj.is_variable_type():
                    if not obj.is_fixed():
                        # TODO: what should we do if it is fixed?
                        fixed_inputs[dest_unit].add(obj)
                        obj.fix(x[i])
                else:
                    repn = generate_standard_repn(obj - x[i])
                    if not repn.is_fixed():
                        # TODO: what should we do if it is fixed?
                        if repn.is_linear() and len(repn.linear_vars) == 1:
                            # fix the value of the single variable
                            val = (0 - repn.constant) / repn.linear_coefs[0]
                            var = repn.linear_vars[0]
                            fixed_inputs[dest_unit].add(var)
                            var.fix(val)
                        else:
                            # TODO: what to do if we're underspecified, make
                            # a model and stuff it with constraints and solve?
                            need_to_solve = True
                            logger.warning("Dest member '%s' of arc '%s' has "
                                "more than one free variable." %
                                (name, arc.name))
                i += 1

            if need_to_solve:
                raise Exception("I haven't figured this out yet")

    def generate_gofx(self, G, tears):
        edge_list = self.idx_to_edge(G)
        gofx = []
        for tear in tears:
            arc = G.edges[edge_list[tear]]["arc"]
            for name, mem in arc.src.iter_vars(with_names=True):
                sf = arc.expanded_block.component("splitfrac")
                if sf is not None:
                    gofx.append(value(mem * sf))
                else:
                    gofx.append(value(mem))
        return gofx

    def generate_weg_lists(self, G, tears):
        edge_list = self.idx_to_edge(G)
        gofx = []
        x = []
        xmin = []
        xmax = []

        for tear in tears:
            arc = G.edges[edge_list[tear]]["arc"]
            src, dest = arc.src, arc.dest
            sf = arc.expanded_block.component("splitfrac")
            for name, mem in src.iter_vars(with_names=True):
                if sf is not None:
                    gofx.append(value(mem * sf))
                else:
                    gofx.append(value(mem))
                try:
                    index = mem.index()
                except AttributeError:
                    index = None
                obj = self.source_dest_peer(arc, name, index)
                val = value(obj)
                x.append(val)
                # TODO: what to do if it doesn't have bound(s)
                try:
                    xmax.append(obj.ub if obj.has_ub() else val)
                    xmin.append(obj.lb if obj.has_lb() else val)
                except AttributeError:
                    # expressions don't have bounds
                    xmax.append(val)
                    xmin.append(val)

        return gofx, x, xmin, xmax

    def cacher(self, key, fcn, *args):
        if key in self.cache:
            return self.cache[key]
        res = fcn(*args)
        self.cache[key] = res
        return res

    def tear_set(self, G):
        key = "tear_set"
        def fcn(G):
            res = self.options[key]
            if res is not None:
                if not self.check_tear_set(G, res):
                    raise ValueError("Tear set found in options is "
                                     "insufficient to solve network")
                self.cache[key] = res
                return res

            method = self.options["select_tear_method"]
            if method == "mip":
                return self.select_tear_mip(G,
                                            self.options["tear_solver"],
                                            self.options["tear_solver_io"],
                                            self.options["tear_solver_options"])
            elif method == "heuristic":
                # tset is the first list in the first return value
                return self.select_tear_heuristic(G)[0][0]
            else:
                raise ValueError("Invalid select_tear_method '%s'" % (method,))
        return self.cacher(key, fcn, G)

    def fixed_inputs(self):
        return self.cacher("fixed_inputs", dict)

    def idx_to_node(self, G):
        """Returns a mapping from indexes to nodes for a graph"""
        return self.cacher("idx_to_node", list, G.nodes)

    def node_to_idx(self, G):
        """Returns a mapping from nodes to indexes for a graph"""
        def fcn(G):
            res = dict()
            i = -1
            for node in G.nodes:
                i += 1
                res[node] = i
            return res
        return self.cacher("node_to_idx", fcn, G)

    def idx_to_edge(self, G):
        """Returns a mapping from indexes to edges for a graph"""
        return self.cacher("idx_to_edge", list, G.edges)

    def edge_to_idx(self, G):
        """Returns a mapping from edges to indexes for a graph"""
        def fcn(G):
            res = dict()
            i = -1
            for edge in G.edges:
                i += 1
                res[edge] = i
            return res
        return self.cacher("edge_to_idx", fcn, G)

    ########################################################################
    #
    # The following code is adapted from graph.py in FOQUS:
    # https://github.com/CCSI-Toolset/FOQUS/blob/master/LICENSE.md
    # It has been modified to use networkx graphs and should be
    # independent of Pyomo or whatever the nodes actually are.
    #
    ########################################################################

    def solve_tear_direct(self, G, order, function, tears, iterLim, tol):
        """
        Use direct substitution to solve tears. If multiple tears are
        given they are solved simultaneously.

        Arguments:
            order           List of lists of order in which to calculate nodes
            tears           List of tear edge indexes
            iterLim         Limit on the number of iterations to run
            tol             Tolerance at which iteration can be stopped

        Returns:
            List of lists of error history, differences between input and
                output values at each iteration.
        """
        hist = [] # error at each iteration in every variable
        itercount = 0 # iteration counter

        if not len(tears):
            # no need to iterate just run the calculations
            self.run_order(G, order, function)
            return hist

        logger.info("Starting Direct tear convergence")

        fixed_outputs = ComponentSet()
        edge_list = self.idx_to_edge(G)

        while True:
            err = self.tear_error(G, tears)
            hist.append(err)

            if numpy.max(numpy.abs(err)) < tol:
                logger.info("Direct converged in %s iterations" % itercount)
                break

            if itercount >= iterLim:
                logger.warning("Direct failed to converge in %s iterations"
                    % iterLim)
                return hist

            for tear in tears:
                # fix everything then call pass values
                arc = G.edges[edge_list[tear]]["arc"]
                for var in arc.source.iter_vars(expr_vars=True, fixed=False):
                    fixed_outputs.add(var)
                    var.fix()
                self.pass_values(arc, fixed_inputs=self.fixed_inputs())
                for var in fixed_outputs:
                    var.free()
                fixed_outputs.clear()

            itercount += 1
            logger.info("Running Direct iteration %s" % itercount)
            self.run_order(G, order, function)

        return hist

    def solve_tear_wegstein(self, G, order, function, tears, iterLim,
        tol, accelMin, accelMax, tearTolType):
        """
        Use Wegstein to solve tears. If multiple tears are given
        they are solved simultaneously.

        Arguments:
            order           List of lists of order in which to calculate nodes
            tears           List of tear edge indexes
            iterLim         Limit on the number of iterations to run
            tol             Tolerance at which iteration can be stopped
            accelMin        Minimum value for Wegstein acceleration factor
            accelMax        Maximum value for Wegstein acceleration factor
            tearTolType     Interpretation of tolerance value, either:
                                "abs" - Absolute tolerance
                                "rel" - Relative tolerance (to bound range)

        Returns:
            List of lists of error history, differences between input and
                output values at each iteration.
        """
        hist = [] # error at each iteration in every variable
        itercount = 0 # iteration counter

        if not len(tears):
            # no need to iterate just run the calculations
            self.run_order(G, order, function)
            return hist

        logger.info("Starting Wegstein tear convergence")

        gofx, x, xmin, xmax = self.generate_weg_lists(G, tears)
        gofx = numpy.array(gofx)
        x = numpy.array(x)
        xmin = numpy.array(xmin)
        xmax = numpy.array(xmax)
        xrng = xmax - xmin

        if tearTolType == "abs":
            err = gofx - x
        elif tearTolType == "rel":
            err = (gofx - x) / xrng
        else:
            raise ValueError("Invalid tearTolType '%s'" % (tearTolType,))
        hist.append(err)

        # check if it's already solved
        if numpy.max(numpy.abs(err)) < tol:
            logger.info("Wegstein converged in %s iterations" % itercount)
            return hist

        # if not solved yet do one direct step
        x_prev = x
        gofx_prev = gofx
        x = gofx
        self.pass_tear_weg(G, tears, gofx)

        while True:
            itercount += 1

            logger.info("Running Wegstein iteration %s" % itercount)
            self.run_order(G, order, function)

            gofx = self.generate_gofx(G, tears)
            gofx = numpy.array(gofx)

            if tearTolType == "abs":
                err = gofx - x
            elif tearTolType == "rel":
                err = (gofx - x) / xrng
            hist.append(err)

            if numpy.max(numpy.abs(err)) < tol:
                logger.info("Wegstein converged in %s iterations" % itercount)
                break

            if itercount > iterLim:
                logger.warning("Wegstein failed to converge in %s iterations"
                    % iterLim)
                return hist

            denom = x - x_prev
            # this will divide by 0 at some points but we handle that below,
            # so ignore division warnings
            old_settings = numpy.seterr(divide='ignore', invalid='ignore')
            slope = numpy.divide((gofx - gofx_prev), denom)
            numpy.seterr(**old_settings)
            # if isnan or isinf then x and x_prev were the same,
            # so just do direct sub for those elements
            slope[numpy.isnan(slope)] = 0.0
            slope[numpy.isinf(slope)] = 0.0
            accel = slope / (slope - 1)
            accel[accel < accelMin] = accelMin
            accel[accel > accelMax] = accelMax
            x_prev = x
            gofx_prev = gofx
            x = accel * x_prev + (1 - accel) * gofx_prev
            self.pass_tear_weg(G, tears, x)

        return hist

    def scc_collect(self, G, excludeEdges=None):
        """
        This is an algorithm for finding strongly connected components (SCCs)
        in a graph. It is based on Tarjan. 1972 Depth-First Search and Linear
        Graph Algorithms, SIAM J. Comput. v1 no. 2 1972

        Returns:
            List of lists of nodes in each SCC
            List of lists of edges in each SCC
            List of lists for order in which to calculate SCCs
        """
        def sc(v, stk, depth, strngComps):
            # recursive sub-function for backtracking
            ndepth[v] = depth
            back[v] = depth
            depth += 1
            stk.append(v)
            for w in adj[v]:
                if ndepth[w] == None:
                    sc(w, stk, depth, strngComps)
                    back[v] = min(back[w], back[v])
                elif w in stk:
                    back[v] = min(back[w], back[v])
            if back[v] == ndepth[v]:
                scomp = []
                while True:
                    w = stk.pop()
                    scomp.append(i2n[w])
                    if w == v:
                        break
                strngComps.append(scomp)
            return depth

        i2n, adj, _ = self.adj_lists(G, excludeEdges=excludeEdges)

        stk        = []  # node stack
        strngComps = []  # list of SCCs
        ndepth     = [None] * G.number_of_nodes()
        back       = [None] * G.number_of_nodes()

        # find the SCCs
        for v in range(G.number_of_nodes()):
            if ndepth[v] == None:
                sc(v, stk, 0, strngComps)

        # Find the rest of the information about SCCs given the node partition
        sccNodes = strngComps
        sccEdges = []
        outEdges = []
        inEdges = []
        for nset in strngComps:
            e, ie, oe = self.sub_graph_edges(G, nset)
            sccEdges.append(e)
            inEdges.append(ie)
            outEdges.append(oe)
        sccOrder = self.scc_calculation_order(sccNodes, inEdges, outEdges)
        return sccNodes, sccEdges, sccOrder

    def scc_calculation_order(self, sccNodes, ie, oe):
        """
        This determines the order in which to do calculations for strongly
        connected components. It is used to help determine the most efficient
        order to solve tear streams. For example, if you have a graph like
        the following, you would want to do tear streams in SCC0 before SCC1
        and SCC2 to prevent extra iterations. This just makes an adjacency
        list with the SCCs as nodes and calls the tree order function.

        SCC0--+-->--SCC1
              |
              +-->--SCC2

        Arguments:
            sccNodes        List of lists of nodes in each SCC
            ie              List of lists of in edges to SCCs
            oe              List of lists of out edged to SCCs

        """
        adj = [] # SCC adjacency list
        adjR = [] # SCC reverse adjacency list
        for i in range(len(sccNodes)):
            adj.append([])
            adjR.append([])

        # build adjacency lists
        done = False
        for i in range(len(sccNodes)):
            for j in range(len(sccNodes)):
                for ine in ie[i]:
                    for oute in oe[j]:
                        if ine == oute:
                            adj[j].append(i)
                            adjR[i].append(j)
                            done = True
                    if done:
                        break
                if done:
                    break
            done = False

        return self.tree_order(adj, adjR)

    def calculation_order(self, G, roots=None, nodes=None):
        """Rely on tree_order to return a calculation order of nodes."""
        tset = self.tear_set(G)
        i2n, adj, adjR = self.adj_lists(G, excludeEdges=tset, nodes=nodes)

        order = []
        if roots is not None:
            node_map = self.node_to_idx(G)
            rootsIndex = []
            for node in roots:
                rootsIndex.append(node_map[node])
        else:
            rootsIndex = None

        orderIndex = self.tree_order(adj, adjR, rootsIndex)

        # convert indexes to actual nodes
        for i in range(len(orderIndex)):
            order.append([])
            for j in range(len(orderIndex[i])):
                order[i].append(i2n[orderIndex[i][j]])

        return order

    def tree_order(self, adj, adjR, roots=None):
        """
        This function determines the ordering of nodes in a directed
        tree. This is a generic function that can operate on any
        given tree represented by the adjaceny and reverse
        adjacency lists. If the adjacency list does not represent
        a tree the results are not valid.

        In the returned order, it is sometimes possible for more
        than one node to be caclulated at once. So a list of lists
        is returned by this function. These represent a bredth
        first search order of the tree. Following the order, all
        nodes that lead to a particular node will be visited
        before it.

        Arguments:
            adj: an adjeceny list for a directed tree. This uses
                generic integer node indexes, not node names from the
                graph itself. This allows this to be used on sub-graphs
                and graps of components more easily.
            adjR: the reverse adjacency list coresponing to adj
            roots: list of node indexes to start from. These do not
                need to be the root nodes of the tree, in some cases
                like when a node changes the changes may only affect
                nodes reachable in the tree from the changed node, in
                the case that roots are supplied not all the nodes in
                the tree may appear in the ordering. If no roots are
                supplied, the roots of the tree are used.
        """
        adjR = copy.deepcopy(adjR)
        for i, l in enumerate(adjR):
            adjR[i] = set(l)

        if roots == None:
            roots = []
            mark = [True] * len(adj) # mark all nodes if no roots specified
            r = [True] * len(adj)
            # no root specified so find roots of tree by marking every
            # successor of every node, since roots have no predecessors
            for sucs in adj:
                for i in sucs:
                    r[i] = False
            # make list of roots
            for i in range(len(r)):
                if r[i]:
                    roots.append(i)
        else:
            # if roots are specified mark descendants
            mark = [False] * len(adj)
            lst = roots
            while len(lst) > 0:
                lst2 = []
                for i in lst:
                    mark[i] = True
                    lst2 += adj[i]
                lst = set(lst2) # remove dupes

        # Now we have list of roots, and roots and their desendants are marked
        ndepth = [None] * len(adj)
        lst = copy.deepcopy(roots)
        order = []
        checknodes = set() # list of candidate nodes for next depth
        for i in roots: # nodes adjacent to roots are candidates
            checknodes.update(adj[i])
        depth = 0

        while len(lst) > 0:
            order.append(lst)
            depth += 1
            lst = [] # nodes to add to the next depth in order
            delSet = set() # nodes to delete from checknodes
            checkUpdate = set() # nodes to add to checknodes
            for i in checknodes:
                if ndepth[i] != None:
                    # This means there is a cycle in the graph
                    # this will lead to nonsense so throw exception
                    raise RuntimeError(
                        "Function tree_order does not work with cycles")
                remSet = set() # to remove from a nodes rev adj list
                for j in adjR[i]:
                    if j in order[depth - 1]:
                        # ancestor already placed
                        remSet.add(j)
                    elif mark[j] == False:
                        # ancestor doesn't descend from root
                        remSet.add(j)
                # delete parents from rev adj list if they were found
                # to be already placed or not in subgraph
                adjR[i] = adjR[i].difference(remSet)
                # if rev adj list is empty, all ancestors
                # have been placed so add node
                if len(adjR[i]) == 0:
                    ndepth[i] = depth
                    lst.append(i)
                    delSet.add(i)
                    checkUpdate.update(adj[i])
            # Delete the nodes that were added from the check set
            checknodes = checknodes.difference(delSet)
            checknodes = checknodes.union(checkUpdate)

        return order

    def check_tear_set(self, G, tset):
        """
        Check whether the specified tear streams are sufficient.
        If the graph minus the tear edges is not a tree then the
        tear set is not sufficient to solve the graph.
        """
        sccNodes, _, _ = self.scc_collect(G, excludeEdges=tset)
        for nodes in sccNodes:
            if len(nodes) > 1:
                return False
        return True

    def select_tear_heuristic(self, G):
        """
        This finds optimal sets of tear edges based on two criteria.
        The primary objective is to minimize the maximum number of
        times any cycle is broken. The seconday criteria is to
        minimize the number of tears. This function uses a branch
        and bound type approach.

        Output:
            List of lists of tear sets. All the tear sets returned
            are equally good there are often a very large number of
            equally good tear sets.

        Improvemnts for the future.
        I think I can imporve the efficency of this, but it is good
        enough for now. Here are some ideas for improvement:
            1) Reduce the number of redundant solutions. It is possible
               to find tears sets [1,2] and [2,1]. I eliminate
               redundent solutions from the results, but they can
               occur and it reduces efficency.
            2) Look at strongly connected components instead of whole
               graph. This would cut back on the size of graph we are
               looking at. The flowsheets are rarely one strongly
               conneted component.
            3) When you add an edge to a tear set you could reduce the
               size of the problem in the branch by only looking at
               strongly connected components with that edge removed.
            4) This returns all equally good optimal tear sets. That
               may not really be necessary. For very large flowsheets,
               there could be an extremely large number of optimial tear
               edge sets.
        """

        def sear(depth, prevY):
            # This is a recursive function for generating tear sets.
            # It selects one edge from a cycle, then calls itself
            # to select an edge from the next cycle.  It is a branch
            # and bound search tree to find best tear sets.

            # The function returns when all cycles are torn, which
            # may be before an edge was selected from each cycle if
            # cycles contain common edges.

            for i in range(len(cycleEdges[depth])):
                # Loop through all the edges in cycle with index depth
                y = list(prevY) # get list of already selected tear stream
                y[cycleEdges[depth][i]] = 1
                # calculate number of times each cycle is torn
                Ay = numpy.dot(A, y)
                maxAy = max(Ay)
                sumY = sum(y)
                if maxAy > upperBound[0]:
                    # breaking a cycle too many times, branch is no good
                    continue
                elif maxAy == upperBound[0] and sumY > upperBound[1]:
                    # too many tears, branch is no good
                    continue
                # Call self at next depth where a cycle is not broken
                if min(Ay) > 0:
                    if maxAy < upperBound[0]:
                        upperBound[0] = maxAy  # most important factor
                        upperBound[1] = sumY   # second most important
                    elif sumY < upperBound[1]:
                        upperBound[1] = sumY
                    # record solution
                    ySet.append([list(y), maxAy, sumY])
                else:
                    for j in range(depth + 1, nr):
                        if Ay[j] == 0:
                            sear(j, y)

        # Get a quick and I think pretty good tear set for upper bound
        tearUB = self.tear_upper_bound(G)

        # Find all the cycles in a graph and make cycle-edge matrix A
        # Rows of A are cycles and columns of A are edges
        # 1 if an edge is in a cycle, 0 otherwise
        A, _, cycleEdges = self.cycle_edge_matrix(G)
        (nr, nc) = A.shape

        if nr == 0:
            # no cycles so we are done
            return [[[]], 0 , 0]

        # Else there are cycles, so find edges to tear
        y_init = [False] * G.number_of_edges() # whether edge j is in tear set
        for j in tearUB:
            # y for initial u.b. solution
            y_init[j] = 1

        Ay_init = numpy.dot(A, y_init) # number of times each loop torn

        # Set two upper bounds. The fist upper bound is on number of times
        # a loop is broken. Second upper bound is on number of tears.
        upperBound = [max(Ay_init), sum(y_init)]

        y_init = [False] * G.number_of_edges() #clear y vector to start search
        ySet = []  # a list of tear sets
        # Three elements are stored in each tear set:
        # 0 = y vector (tear set), 1 = max(Ay), 2 = sum(y)

        # Call recursive function to find tear sets
        sear(0, y_init)

        # Screen tear sets found
        # A set can be recorded before upper bound is updated so we can
        # just throw out sets with objectives higher than u.b.
        deleteSet = []  # vector of tear set indexes to delete
        for i in range(len(ySet)):
            if ySet[i][1] > upperBound[0]:
                deleteSet.append(i)
            elif ySet[i][1] == upperBound[0] and ySet[i][2] > upperBound[1]:
                deleteSet.append(i)
        for i in reversed(deleteSet):
            del ySet[i]

        # Check for duplicates and delete them
        deleteSet = []
        for i in range(len(ySet) - 1):
            if i in deleteSet:
                continue
            for j in range(i + 1, len(ySet)):
                if j in deleteSet:
                    continue
                for k in range(len(y_init)):
                    eq = True
                    if ySet[i][0][k] != ySet[j][0][k]:
                        eq = False
                        break
                if eq == True:
                    deleteSet.append(j)
        for i in reversed(sorted(deleteSet)):
            del ySet[i]

        # Turn the binary y vectors into lists of edge indexes
        es = []
        for y in ySet:
            edges = []
            for i in range(len(y[0])):
                if y[0][i] == 1:
                    edges.append(i)
            es.append(edges)

        return es, upperBound[0], upperBound[1]

    def tear_upper_bound(self, G):
        """
        This function quickly finds a sub-optimal set of tear
        edges. This serves as an inital upperbound when looking
        for an optimal tear set. Having an inital upper bound
        improves efficiency.

        This works by constructing a search tree and just makes a
        tear set out of all the back edges.
        """

        def cyc(node, depth):
            # this is a recursive function
            depths[node] = depth
            depth += 1
            for suc in G.successors(node):
                if depths[suc] is None:
                    parents[suc] = node
                    cyc(suc, depth)
                elif depths[suc] < depths[node]:
                    # found a back edge, add to tear set
                    tearSet.append(edge_list.index((node, suc)))

        tearSet = []  # list of back/tear edges
        edge_list = self.idx_to_edge(G)
        depths = {}
        parents = {}

        for node in G.nodes:
            depths[node]  = None
            parents[node]  = None

        for node in G.nodes:
            if depths[node] is None:
                cyc(node, 0)

        return tearSet

    def sub_graph_edges(self, G, nodes):
        """
        This function returns a list of edge indexes that are
        included in a subgraph given by a list of nodes.

        Returns:
            List of edges in the subgraph
            List of edges starting outside the subgraph and ending inside
            List of edges starting inside the subgraph and ending outside
        """
        e = []   # edges that connect two nodes in the subgraph
        ie = []  # in edges
        oe = []  # out edges
        edge_list = self.idx_to_edge(G)
        for i in range(G.number_of_edges()):
            src, dest = edge_list[i]
            if src in nodes:
                if dest in nodes:
                    # it's in the sub graph
                    e.append(i)
                else:
                    # it's an out edge of the subgraph
                    oe.append(i)
            elif dest in nodes:
                #its a in edge of the subgraph
                ie.append(i)
        return e, ie, oe

    def cycle_edge_matrix(self, G):
        """
        Return a cycle-edge incidence matrix, a list of list of nodes in
        each cycle, and a list of list of edge indexes in each cycle.
        """
        cycles, cycleEdges = self.all_cycles(G) # call cycle finding algorithm

        # Create empty incidence matrix and then fill it out
        ceMat = numpy.zeros((len(cycleEdges), G.number_of_edges()),
                            dtype=numpy.dtype(int))
        for i in range(len(cycleEdges)):
            for e in cycleEdges[i]:
                ceMat[i, e] = 1

        return ceMat, cycles, cycleEdges

    def all_cycles(self, G):
        """
        This function finds all the cycles in a directed graph.
        The algorithm is based on Tarjan 1973 Enumeration of the
        elementary circuits of a directed graph, SIAM J. Comput. v3 n2 1973.

        Returns:
            List of lists of nodes in each cycle
            List of lists of edge indexes in each cycle
        """

        def backtrack(v):
            # sub-function recursive part
            f = False
            pointStack.append(v)
            mark[v] = True
            markStack.append(v)
            sis = list(adj[v])

            for si in sis:
                # iterate over successor indexes
                if si < ni:
                    adj[v].remove(si)
                elif si == ni:
                    f = True
                    cycles.append(list(pointStack))
                elif not mark[si]:
                    g = backtrack(si)
                    f = f or g

            if f == True:
                while markStack[-1] != v:
                    u = markStack.pop()
                    mark[u] = False
                markStack.pop()
                mark[v] = False

            pointStack.pop()
            return f

        i2n, adj, _ = self.adj_lists(G)
        pointStack  = [] # node stack
        markStack = [] # nodes that have been marked
        cycles = [] # list of cycles found
        mark = [False] * G.number_of_nodes() # if a node is marked

        for ni in range(G.number_of_nodes()):
            # iterate over node indexes
            backtrack(ni)
            while len(markStack) > 0:
                i = markStack.pop()
                mark[i] = False

        # Turn node indexes back into nodes
        for cycle in cycles:
            for i in range(len(cycle)):
                cycle[i] = i2n[cycle[i]]

        # Now find list of edges in the cycle
        edge_map = self.edge_to_idx(G)
        cycleEdges = []
        for cyc in cycles:
            ecyc = []
            for i in range(len(cyc) - 1):
                ecyc.append(edge_map[(cyc[i], cyc[i+1])])
            ecyc.append(edge_map[(cyc[-1], cyc[0])]) # edge from end to start
            cycleEdges.append(ecyc)
        return cycles, cycleEdges

    def adj_lists(self, G, excludeEdges=None, nodes=None):
        """
        Returns an adjacency matrix and a reverse adjacency matrix
        of node indexes for a DiGraph. Pass a list of edge indexes to
        ignore certain edges when considering neighbors. Pass a list of
        nodes to only form the adjacencies from those nodes.
        """
        adj = []
        adjR = []

        exclude = set()
        if excludeEdges is not None:
            edge_list = self.idx_to_edge(G)
            for ei in excludeEdges:
                exclude.add(edge_list[ei])

        if nodes is None:
            nodes = self.idx_to_node(G)

        # we might not be including every node in these lists, so we need
        # custom maps to get between indexes and nodes
        i = -1
        i2n = [None] * len(nodes)
        n2i = dict()
        for node in nodes:
            i += 1
            n2i[node] = i
            i2n[i] = node

        i = -1
        for node in nodes:
            i += 1
            adj.append([])
            adjR.append([])
            for suc in G.successors(node):
                if suc in nodes and (node, suc) not in exclude:
                    adj[i].append(n2i[suc])
            for pre in G.predecessors(node):
                if pre in nodes and (pre, node) not in exclude:
                    adjR[i].append(n2i[pre])

        return i2n, adj, adjR
