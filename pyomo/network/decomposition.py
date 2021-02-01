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
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import Constraint, value, Objective, Var, ConcreteModel, \
    Binary, minimize, Expression
from pyomo.common.collections import ComponentSet, ComponentMap, Options
from pyomo.core.expr.current import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from six import iteritems

from pyomo.common.dependencies import (
    networkx as nx, networkx_available,
    numpy, numpy_available,
)

imports_available = networkx_available & numpy_available

logger = logging.getLogger('pyomo.network')

class SequentialDecomposition(FOQUSGraph):
    """
    A sequential decomposition tool for Pyomo Network models

    The following parameters can be set upon construction of this class
    or via the `options` attribute.

    Parameters
    ----------
        graph: `MultiDiGraph`
            A networkx graph representing the model to be solved.

            `default=None (will compute it)`

        tear_set: `list`
            A list of indexes representing edges to be torn. Can be set with
            a list of edge tuples via set_tear_set.

            `default=None (will compute it)`

        select_tear_method: `str`
            Which method to use to select a tear set, either "mip" or
            "heuristic".

            `default="mip"`

        run_first_pass: `bool`
            Boolean indicating whether or not to run through network before
            running the tear stream convergence procedure.

            `default=True`

        solve_tears: `bool`
            Boolean indicating whether or not to run iterations to converge
            tear streams.

            `default=True`

        guesses: `ComponentMap`
            ComponentMap of guesses to use for first pass
            (see set_guesses_for method).

            `default=ComponentMap()`

        default_guess: `float`
            Value to use if a free variable has no guess.

            `default=None`

        almost_equal_tol: `float`
            Difference below which numbers are considered equal when checking
            port value agreement.

            `default=1.0E-8`

        log_info: `bool`
            Set logger level to INFO during run.

            `default=False`

        tear_method: `str`
            Method to use for converging tear streams, either "Direct" or
            "Wegstein".

            `default="Direct"`

        iterLim: `int`
            Limit on the number of tear iterations.

            `default=40`

        tol: `float`
            Tolerance at which to stop tear iterations.

            `default=1.0E-5`

        tol_type: `str`
            Type of tolerance value, either "abs" (absolute) or
            "rel" (relative to current value).

            `default="abs"`

        report_diffs: `bool`
            Report the matrix of differences across tear streams for
            every iteration.

            `default=False`

        accel_min: `float`
            Min value for Wegstein acceleration factor.

            `default=-5`

        accel_max: `float`
            Max value for Wegstein acceleration factor.

            `default=0`

        tear_solver: `str`
            Name of solver to use for select_tear_mip.

            `default="cplex"`

        tear_solver_io: `str`
            Solver IO keyword for the above solver.

            `default=None`

        tear_solver_options: `dict`
            Keyword options to pass to solve method.

            `default={}`
    """

    def __init__(self, **kwds):
        """Pass kwds to update the options attribute after setting defaults"""
        self.cache = {}
        options = self.options = Options()
        # defaults
        options["graph"] = None
        options["tear_set"] = None
        options["select_tear_method"] = "mip"
        options["run_first_pass"] = True
        options["solve_tears"] = True
        options["guesses"] = ComponentMap()
        options["default_guess"] = None
        options["almost_equal_tol"] = 1.0E-8
        options["log_info"] = False
        options["tear_method"] = "Direct"
        options["iterLim"] = 40
        options["tol"] = 1.0E-5
        options["tol_type"] = "abs"
        options["report_diffs"] = False
        options["accel_min"] = -5
        options["accel_max"] = 0
        options["tear_solver"] = "cplex"
        options["tear_solver_io"] = None
        options["tear_solver_options"] = {}

        options.update(kwds)

    def set_guesses_for(self, port, guesses):
        """
        Set the guesses for the given port

        These guesses will be checked for all free variables that are
        encountered during the first pass run. If a free variable has
        no guess, its current value will be used. If its current value
        is None, the default_guess option will be used. If that is None,
        an error will be raised.

        All port variables that are downstream of a non-tear edge will
        already be fixed. If there is a guess for a fixed variable, it
        will be silently ignored.

        The guesses should be a dict that maps the following:

            Port Member Name -> Value

        Or, for indexed members, multiple dicts that map:

            Port Member Name -> Index -> Value

        For extensive members, "Value" must be a list of tuples of the
        form (arc, value) to guess a value for the expanded variable
        of the specified arc. However, if the arc connecting this port
        is a 1-to-1 arc with its peer, then there will be no expanded
        variable for the single arc, so a regular "Value" should be
        provided.

        This dict cannot be used to pass guesses for variables within
        expression type members. Guesses for those variables must be
        assigned to the variable's current value before calling run.

        While this method makes things more convenient, all it does is:

            `self.options["guesses"][port] = guesses`
        """
        self.options["guesses"][port] = guesses

    def set_tear_set(self, tset):
        """
        Set a custom tear set to be used when running the decomposition

        The procedure will use this custom tear set instead of finding
        its own, thus it can save some time. Additionally, this will be
        useful for knowing which edges will need guesses.

        Arguments
        ---------
            tset
                A list of Arcs representing edges to tear

        While this method makes things more convenient, all it does is:

            `self.options["tear_set"] = tset`
        """
        self.options["tear_set"] = tset

    def tear_set_arcs(self, G, method="mip", **kwds):
        """
        Call the specified tear selection method and return a list
        of arcs representing the selected tear edges.

        The kwds will be passed to the method.
        """
        if method == "mip":
            tset = self.select_tear_mip(G, **kwds)
        elif method == "heuristic":
            # tset is the first list in the first return value
            tset = self.select_tear_heuristic(G, **kwds)[0][0]
        else:
            raise ValueError("Invalid method '%s'" % (method,))

        return self.indexes_to_arcs(G, tset)

    def indexes_to_arcs(self, G, lst):
        """
        Converts a list of edge indexes to the corresponding Arcs

        Arguments
        ---------
            G
                A networkx graph corresponding to lst
            lst
                A list of edge indexes to convert to tuples

        Returns:
            A list of arcs
        """
        edge_list = self.idx_to_edge(G)
        res = []
        for ei in lst:
            edge = edge_list[ei]
            res.append(G.edges[edge]["arc"])
        return res

    def run(self, model, function):
        """
        Compute a Pyomo Network model using sequential decomposition

        Arguments
        ---------
            model
                A Pyomo model
            function
                A function to be called on each block/node in the network
        """
        if self.options["log_info"]:
            old_log_level = logger.level
            logger.setLevel(logging.INFO)

        self.cache.clear()

        try:
            return self._run_impl(model, function)
        finally:
            # Cleanup
            self.cache.clear()

            if self.options["log_info"]:
                logger.setLevel(old_log_level)


    def _run_impl(self, model, function):
        start = time.time()
        logger.info("Starting Sequential Decomposition")

        G = self.options["graph"]
        if G is None:
            G = self.create_graph(model)

        tset = self.tear_set(G)

        if self.options["run_first_pass"]:
            logger.info("Starting first pass run of network")
            order = self.calculation_order(G)
            self.run_order(G, order, function, tset, use_guesses=True)

        if not self.options["solve_tears"] or not len(tset):
            # Not solving tears, we're done
            end = time.time()
            logger.info("Finished Sequential Decomposition in %.2f seconds" %
                (end - start))
            return

        logger.info("Starting tear convergence procedure")

        sccNodes, sccEdges, sccOrder, outEdges = self.scc_collect(G)

        for lev in sccOrder:
            for sccIndex in lev:
                order = self.calculation_order(G, nodes=sccNodes[sccIndex])

                # only pass tears that are part of this SCC
                tears = []
                for ei in tset:
                    if ei in sccEdges[sccIndex]:
                        tears.append(ei)

                kwds = dict(G=G, order=order, function=function, tears=tears,
                    iterLim=self.options["iterLim"], tol=self.options["tol"],
                    tol_type=self.options["tol_type"],
                    report_diffs=self.options["report_diffs"],
                    outEdges=outEdges[sccIndex])

                tear_method = self.options["tear_method"]

                if tear_method == "Direct":
                    self.solve_tear_direct(**kwds)

                elif tear_method == "Wegstein":
                    kwds["accel_min"] = self.options["accel_min"]
                    kwds["accel_max"] = self.options["accel_max"]
                    self.solve_tear_wegstein(**kwds)

                else:
                    raise ValueError(
                        "Invalid tear_method '%s'" % (tear_method,))

        end = time.time()
        logger.info("Finished Sequential Decomposition in %.2f seconds" %
            (end - start))

    def run_order(self, G, order, function, ignore=None, use_guesses=False):
        """
        Run computations in the order provided by calling the function

        Arguments
        ---------
            G
                A networkx graph corresponding to order
            order
                The order in which to run each node in the graph
            function
                The function to be called on each block/node
            ignore
                Edge indexes to ignore when passing values
            use_guesses
                If True, will check the guesses dict when fixing
                free variables before calling function
        """
        fixed_inputs = self.fixed_inputs()
        fixed_outputs = ComponentSet()
        edge_map = self.edge_to_idx(G)
        guesses = self.options["guesses"]
        default = self.options["default_guess"]
        for lev in order:
            for unit in lev:
                if unit not in fixed_inputs:
                    fixed_inputs[unit] = ComponentSet()
                fixed_ins = fixed_inputs[unit]

                # make sure all inputs are fixed
                for port in unit.component_data_objects(Port):
                    if not len(port.sources()):
                        continue
                    if use_guesses and port in guesses:
                        self.load_guesses(guesses, port, fixed_ins)
                    self.load_values(port, default, fixed_ins, use_guesses)

                function(unit)

                # free the inputs that were not already fixed
                for var in fixed_ins:
                    var.free()
                fixed_ins.clear()

                # pass the values downstream for all outlet ports
                for port in unit.component_data_objects(Port):
                    dests = port.dests()
                    if not len(dests):
                        continue
                    for var in port.iter_vars(expr_vars=True, fixed=False):
                        fixed_outputs.add(var)
                        var.fix()
                    for arc in dests:
                        arc_map = self.arc_to_edge(G)
                        if edge_map[arc_map[arc]] not in ignore:
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
        src, dest = arc.src, arc.dest
        dest_unit = dest.parent_block()
        eq_tol = self.options["almost_equal_tol"]

        if dest_unit not in fixed_inputs:
            fixed_inputs[dest_unit] = ComponentSet()

        sf = eblock.component("splitfrac")
        if sf is not None and not sf.is_fixed():
            # fix the splitfrac if it has a current value or else error
            if sf.value is not None:
                fixed_inputs[dest_unit].add(sf)
                sf.fix()
            else:
                raise RuntimeError(
                    "Found free splitfrac for arc '%s' with no current value. "
                    "Please use the set_split_fraction method on its source "
                    "port to set this value before expansion, or set its value "
                    "manually if expansion has already occured." % arc.name)
        elif sf is None:
            # if there is no splitfrac, but we have extensive members, then we
            # need to manually set the evar values because there will be no
            # *_split constraints on the eblock, so it is up to us to set it
            # TODO: what if there is no splitfrac, but it's missing because
            # there's only 1 variable per port so it was simplified out?
            # How do we specify the downstream evars? If we assume that the
            # user's function will satisfy the *_outsum constraint before
            # returning, then the evars would at least be specified such that
            # they satisfy the total sum constraint. But I would think we don't
            # want to rely on the user calling solve on their unit before
            # returning, especially since the outsum constraint was auto
            # generated and not one they made themselves.
            # Potential Solution: allow the user to specify a splitfrac
            # (via set_split_fraction or something else) that will be used here
            # and is only relevant to this SM, and if they didn't specify
            # anything, throw an error.
            for name, mem in iteritems(src.vars):
                if not src.is_extensive(name):
                    continue
                evar = eblock.component(name)
                if evar is None:
                    continue
                if len(src.dests()) > 1:
                    raise Exception(
                        "This still needs to be figured out (arc '%s')" %
                        arc.name)
                # TODO: for now we know it's obvious what to do if there is
                # only 1 destination
                if mem.is_indexed():
                    evars = [(evar[i], i) for i in evar]
                else:
                    evars = [(evar, None)]
                for evar, idx in evars:
                    fixed_inputs[dest_unit].add(evar)
                    val = value(mem[idx] if mem.is_indexed() else mem)
                    # val are numpy.float64; coerce val back to float
                    evar.fix(float(val))

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
                if abs(value(con.lower) - repn.constant) > eq_tol:
                    raise RuntimeError(
                        "Found connected ports '%s' and '%s' both with fixed "
                        "but different values (by > %s) for constraint '%s'" %
                        (src, dest, eq_tol, con.name))
                continue
            if not (repn.is_linear() and len(repn.linear_vars) == 1):
                raise RuntimeError(
                    "Constraint '%s' had more than one free variable when "
                    "trying to pass a value to its destination. Please fix "
                    "more variables before passing across this arc." % con.name)
            # fix the value of the single variable to satisfy the constraint
            # con.lower is usually a NumericConstant but call value on it
            # just in case it is something else
            val = (value(con.lower) - repn.constant) / repn.linear_coefs[0]
            var = repn.linear_vars[0]
            fixed_inputs[dest_unit].add(var)
            # val are numpy.float64; coerce val back to float
            var.fix(float(val))

    def pass_single_value(self, port, name, member, val, fixed):
        """
        Fix the value of the port member and add it to the fixed set.
        If the member is an expression, appropriately fix the value of
        its free variable. Error if the member is already fixed but
        different from val, or if the member has more than one free
        variable."
        """
        eq_tol = self.options["almost_equal_tol"]
        if member.is_fixed():
            if abs(value(member) - val) > eq_tol:
                raise RuntimeError(
                    "Member '%s' of port '%s' is already fixed but has a "
                    "different value (by > %s) than what is being passed to it"
                    % (name, port.name, eq_tol))
        elif member.is_expression_type():
            repn = generate_standard_repn(member - val)
            if repn.is_linear() and len(repn.linear_vars) == 1:
                # fix the value of the single variable
                fval = (0 - repn.constant) / repn.linear_coefs[0]
                var = repn.linear_vars[0]
                fixed.add(var)
                # val are numpy.float64; coerce val back to float
                var.fix(float(fval))
            else:
                raise RuntimeError(
                    "Member '%s' of port '%s' had more than "
                    "one free variable when trying to pass a value "
                    "to it. Please fix more variables before passing "
                    "to this port." % (name, port.name))
        else:
            fixed.add(member)
            # val are numpy.float64; coerce val back to float
            member.fix(float(val))

    def load_guesses(self, guesses, port, fixed):
        srcs = port.sources()
        for name, mem in iteritems(port.vars):
            try:
                entry = guesses[port][name]
            except KeyError:
                continue

            if isinstance(entry, dict):
                itr = [(mem[k], entry[k], k) for k in entry]
            elif mem.is_indexed():
                raise TypeError(
                    "Guess for indexed member '%s' in port '%s' must map to a "
                    "dict of indexes" % (name, port.name))
            else:
                itr = [(mem, entry, None)]

            for var, entry, idx in itr:
                if var.is_fixed():
                    # silently ignore vars already fixed
                    continue
                has_evars = False
                if port.is_extensive(name):
                    for arc, val in entry:
                        if arc not in srcs:
                            raise ValueError(
                                "Found a guess for extensive member '%s' on "
                                "port '%s' using arc '%s' that is not a source "
                                "of this port" % (name, port.name, arc.name))
                        evar = arc.expanded_block.component(name)
                        if evar is None:
                            # no evars, 1-to-1 arc
                            break
                        has_evars = True
                        # even if idx is None, we know evar is a Var and
                        # indexing by None into SimpleVars returns itself
                        evar = evar[idx]
                        if evar.is_fixed():
                            # silently ignore vars already fixed
                            continue
                        fixed.add(evar)
                        evar.fix(float(val))
                if not has_evars:
                    # the only NumericValues in Pyomo that return True
                    # for is_fixed are expressions and variables
                    if var.is_expression_type():
                        raise ValueError(
                            "Cannot provide guess for expression type member "
                            "'%s%s' of port '%s', must set current value of "
                            "variables within expression" % (
                                name,
                                ("[%s]" % str(idx)) if mem.is_indexed() else "",
                                port.name))
                    else:
                        fixed.add(var)
                        var.fix(float(entry))

    def load_values(self, port, default, fixed, use_guesses):
        sources = port.sources()
        for name, index, obj in port.iter_vars(fixed=False, names=True):
            evars = None
            if port.is_extensive(name):
                # collect evars if there are any
                evars = [arc.expanded_block.component(name) for arc in sources]
                if evars[0] is None:
                    # no evars, so this arc is 1-to-1
                    evars = None
                else:
                    try:
                        # index into them if necessary, now that
                        # we know they are not None
                        for j in range(len(evars)):
                            evars[j] = evars[j][index]
                    except AttributeError:
                        pass
            if evars is not None:
                for evar in evars:
                    if evar.is_fixed():
                        continue
                    self.check_value_fix(port, evar, default, fixed,
                        use_guesses, extensive=True)
                # now all evars should be fixed so combine them
                # and fix the value of the extensive port member
                self.combine_and_fix(port, name, obj, evars, fixed)
            else:
                if obj.is_expression_type():
                    for var in identify_variables(obj, include_fixed=False):
                        self.check_value_fix(port, var, default, fixed,
                            use_guesses)
                else:
                    self.check_value_fix(port, obj, default, fixed,
                        use_guesses)

    def check_value_fix(self, port, var, default, fixed, use_guesses,
            extensive=False):
        """
        Try to fix the var at its current value or the default, else error
        """
        val = None
        if var.value is not None:
            val = var.value
        elif default is not None:
            val = default

        if val is None:
            raise RuntimeError(
                "Encountered a free inlet %svariable '%s' %s port '%s' with no "
                "%scurrent value, or default_guess option, while attempting "
                "to compute the unit." % (
                    "extensive " if extensive else "",
                    var.name,
                    ("on", "to")[int(extensive)],
                    port.name,
                    "guess, " if use_guesses else ""))

        fixed.add(var)
        var.fix(float(val))

    def combine_and_fix(self, port, name, obj, evars, fixed):
        """
        For an extensive port member, combine the values of all
        expanded variables and fix the port member at their sum.
        Assumes that all expanded variables are fixed.
        """
        assert all(evar.is_fixed() for evar in evars)
        total = sum(value(evar) for evar in evars)
        self.pass_single_value(port, name, obj, total, fixed)

    def source_dest_peer(self, arc, name, index=None):
        """
        Return the object that is the peer to the source port's member.
        This is either the destination port's member, or the variable
        on the arc's expanded block for Extensive properties. This will
        return the appropriate index of the peer.
        """
        # check the rule on source but dest should be the same
        if arc.src.is_extensive(name):
            evar = arc.expanded_block.component(name)
            if evar is not None:
                # 1-to-1 arcs don't make evar because they're an equality
                return evar[index]
        mem = arc.dest.vars[name]
        if mem.is_indexed():
            return mem[index]
        else:
            return mem

    def create_graph(self, model):
        """
        Returns a networkx MultiDiGraph of a Pyomo network model

        The nodes are units and the edges follow Pyomo Arc objects. Nodes
        that get added to the graph are determined by the parent blocks
        of the source and destination Ports of every Arc in the model.
        Edges are added for each Arc using the direction specified by
        source and destination. All Arcs in the model will be used whether
        or not they are active (since this needs to be done after expansion),
        and they all need to be directed.
        """
        G = nx.MultiDiGraph()

        for arc in model.component_data_objects(Arc):
            if not arc.directed:
                raise ValueError("All Arcs must be directed when creating "
                                 "a graph for a model. Found undirected "
                                 "Arc: '%s'" % arc.name)
            if arc.expanded_block is None:
                raise ValueError("All Arcs must be expanded when creating "
                                 "a graph for a model. Found unexpanded "
                                 "Arc: '%s'" % arc.name)
            src, dest = arc.src.parent_block(), arc.dest.parent_block()
            G.add_edge(src, dest, arc=arc)

        return G

    def select_tear_mip_model(self, G):
        """
        Generate a model for selecting tears from the given graph

        Returns
        -------
            model
            bin_list
                A list of the binary variables representing each edge,
                indexed by the edge index of the graph
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

        return model, bin_list

    def select_tear_mip(self, G, solver, solver_io=None, solver_options={}):
        """
        This finds optimal sets of tear edges based on two criteria.
        The primary objective is to minimize the maximum number of
        times any cycle is broken. The seconday criteria is to
        minimize the number of tears.

        This function creates a MIP problem in Pyomo with a doubly
        weighted objective and solves it with the solver arguments.
        """
        model, bin_list = self.select_tear_mip_model(G)

        from pyomo.environ import SolverFactory
        opt = SolverFactory(solver, solver_io=solver_io)
        if not opt.available(exception_flag=False):
            raise ValueError("Solver '%s' (solver_io=%r) is not available, please pass a "
                             "different solver" % (solver, solver_io))
        opt.solve(model, **solver_options)

        # collect final list by adding every edge with a "True" binary var
        tset = []
        for i in range(len(bin_list)):
            if bin_list[i].value == 1:
                tset.append(i)

        return tset

    def compute_err(self, svals, dvals, tol_type):
        """Compute the diff between svals and dvals for the given tol_type"""
        if tol_type not in ("abs", "rel"):
            raise ValueError("Invalid tol_type '%s'" % (tol_type,))

        diff = svals - dvals
        if tol_type == "abs":
            err = diff
        else:
            # relative: divide by current value of svals
            old_settings = numpy.seterr(divide='ignore', invalid='ignore')
            err = diff / svals
            numpy.seterr(**old_settings)
            # isnan means 0/0 so diff is 0
            err[numpy.isnan(err)] = 0
            # isinf means diff/0, so just use the diff
            if any(numpy.isinf(err)):
                for i in range(len(err)):
                    if numpy.isinf(err[i]):
                        err[i] = diff[i]

        return err

    def tear_diff_direct(self, G, tears):
        """
        Returns numpy arrays of values for src and dest members
        for all edges in the tears list of edge indexes.
        """
        svals = []
        dvals = []
        edge_list = self.idx_to_edge(G)
        for tear in tears:
            arc = G.edges[edge_list[tear]]["arc"]
            src, dest = arc.src, arc.dest
            sf = arc.expanded_block.component("splitfrac")
            for name, index, mem in src.iter_vars(names=True):
                if src.is_extensive(name) and sf is not None:
                    # TODO: same as above, what if there's no splitfrac
                    svals.append(value(mem * sf))
                else:
                    svals.append(value(mem))
                dvals.append(value(self.source_dest_peer(arc, name, index)))
        svals = numpy.array(svals)
        dvals = numpy.array(dvals)
        return svals, dvals

    def pass_edges(self, G, edges):
        """Call pass values for a list of edge indexes"""
        fixed_outputs = ComponentSet()
        edge_list = self.idx_to_edge(G)
        for ei in edges:
            arc = G.edges[edge_list[ei]]["arc"]
            for var in arc.src.iter_vars(expr_vars=True, fixed=False):
                fixed_outputs.add(var)
                var.fix()
            self.pass_values(arc, self.fixed_inputs())
            for var in fixed_outputs:
                var.free()
            fixed_outputs.clear()


    def pass_tear_direct(self, G, tears):
        """Pass values across all tears in the given tear set"""
        fixed_outputs = ComponentSet()
        edge_list = self.idx_to_edge(G)

        for tear in tears:
            # fix everything then call pass values
            arc = G.edges[edge_list[tear]]["arc"]
            for var in arc.src.iter_vars(expr_vars=True, fixed=False):
                fixed_outputs.add(var)
                var.fix()
            self.pass_values(arc, fixed_inputs=self.fixed_inputs())
            for var in fixed_outputs:
                var.free()
            fixed_outputs.clear()

    def pass_tear_wegstein(self, G, tears, x):
        """
        Set the destination value of all tear edges to
        the corresponding value in the numpy array x.
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

            for name, index, mem in src.iter_vars(names=True):
                peer = self.source_dest_peer(arc, name, index)
                self.pass_single_value(dest, name, peer, x[i],
                    fixed_inputs[dest_unit])
                i += 1

    def generate_gofx(self, G, tears):
        edge_list = self.idx_to_edge(G)
        gofx = []
        for tear in tears:
            arc = G.edges[edge_list[tear]]["arc"]
            src = arc.src
            sf = arc.expanded_block.component("splitfrac")
            for name, index, mem in src.iter_vars(names=True):
                if src.is_extensive(name) and sf is not None:
                    # TODO: same as above, what if there's no splitfrac
                    gofx.append(value(mem * sf))
                else:
                    gofx.append(value(mem))
        gofx = numpy.array(gofx)
        return gofx

    def generate_first_x(self, G, tears):
        edge_list = self.idx_to_edge(G)
        x = []
        for tear in tears:
            arc = G.edges[edge_list[tear]]["arc"]
            for name, index, mem in arc.src.iter_vars(names=True):
                peer = self.source_dest_peer(arc, name, index)
                x.append(value(peer))
        x = numpy.array(x)
        return x

    def cacher(self, key, fcn, *args):
        if key in self.cache:
            return self.cache[key]
        res = fcn(*args)
        self.cache[key] = res
        return res

    def tear_set(self, G):
        key = "tear_set"
        def fcn(G):
            tset = self.options[key]
            if tset is not None:
                arc_map = self.arc_to_edge(G)
                edge_map = self.edge_to_idx(G)
                res = []
                for arc in tset:
                    res.append(edge_map[arc_map[arc]])
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

    def arc_to_edge(self, G):
        """Returns a mapping from arcs to edges for a graph"""
        def fcn(G):
            res = ComponentMap()
            for edge in G.edges:
                arc = G.edges[edge]["arc"]
                res[arc] = edge
            return res
        return self.cacher("arc_to_edge", fcn, G)

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
