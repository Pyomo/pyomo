#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

##############################################################################
# The methods in this module (file) were derived from the
# `foqus_lib/framework/graph/graph.py` module in the FOQUS package
# (https://github.com/CCSI-Toolset/FOQUS),
# commit hash b28f9cf086b1cfa3d771ffbba014c4bfc15c27b8.
#
# FOQUS License Agreement
#
# Foqus Copyright (c) 2012 - 2018, by the software owners: Oak Ridge Institute
# for Science and Education (ORISE), Los Alamos National Security, LLC.,
# Lawrence Livermore National Security, LLC., The Regents of the University of
# California, through Lawrence Berkeley National Laboratory, Battelle Memorial
# Institute, Pacific Northwest Division through Pacific Northwest National
# Laboratory, Carnegie Mellon University, West Virginia University, Boston
# University, the Trustees of Princeton University, The University of Texas at
# Austin, URS Energy & Construction, Inc., et al. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#           notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#           notice, this list of conditions and the following disclaimer in
#           the documentation and/or other materials provided with the
#           distribution.
#
#     3. Neither the name of the Carbon Capture Simulation Initiative, U.S.
#           Dept. of Energy, the National Energy Technology Laboratory, Oak
#           Ridge Institute for Science and Education (ORISE), Los Alamos
#           National Security, LLC., Lawrence Livermore National Security,
#           LLC., the University of California, Lawrence Berkeley National
#           Laboratory, Battelle Memorial Institute, Pacific Northwest
#           National Laboratory, Carnegie Mellon University, West Virginia
#           University, Boston University, the Trustees of Princeton
#           University, the University of Texas at Austin, URS Energy &
#           Construction, Inc., nor the names of its contributors may be used
#           to endorse or promote products derived from this software without
#           specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You are under no obligation whatsoever to provide any bug fixes, patches, or
# upgrades to the features, functionality or performance of the source code
# ("Enhancements") to anyone; however, if you choose to make your Enhancements
# available either publicly, or directly to Lawrence Berkeley National
# Laboratory, without imposing a separate written license agreement for such
# Enhancements, then you hereby grant the following license: a non-exclusive,
# royalty-free perpetual license to install, use, modify, prepare derivative
# works, incorporate into other computer software, distribute, and sublicense
# such enhancements or derivative works thereof, in binary and source code
# form.
##############################################################################

import copy, logging

try:
    import numpy
except ImportError:
    pass

logger = logging.getLogger('pyomo.network')


class FOQUSGraph(object):
    def solve_tear_direct(self, G, order, function, tears, outEdges, iterLim,
            tol, tol_type, report_diffs):
        """
        Use direct substitution to solve tears. If multiple tears are
        given they are solved simultaneously.

        Arguments
        ---------
            order
                List of lists of order in which to calculate nodes
            tears
                List of tear edge indexes
            iterLim
                Limit on the number of iterations to run
            tol
                Tolerance at which iteration can be stopped

        Returns
        -------
            list
                List of lists of diff history, differences between input and
                output values at each iteration
        """
        hist = [] # diff at each iteration in every variable

        if not len(tears):
            # no need to iterate just run the calculations
            self.run_order(G, order, function, tears)
            return hist

        logger.info("Starting Direct tear convergence")

        ignore = tears + outEdges
        itercount = 0

        while True:
            svals, dvals = self.tear_diff_direct(G, tears)
            err = self.compute_err(svals, dvals, tol_type)
            hist.append(err)

            if report_diffs:
                print("Diff matrix:\n%s" % err)

            if numpy.max(numpy.abs(err)) < tol:
                break

            if itercount >= iterLim:
                logger.warning("Direct failed to converge in %s iterations"
                    % iterLim)
                return hist

            self.pass_tear_direct(G, tears)

            itercount += 1
            logger.info("Running Direct iteration %s" % itercount)
            self.run_order(G, order, function, ignore)

        self.pass_edges(G, outEdges)

        logger.info("Direct converged in %s iterations" % itercount)

        return hist

    def solve_tear_wegstein(self, G, order, function, tears, outEdges, iterLim,
        tol, tol_type, report_diffs, accel_min, accel_max):
        """
        Use Wegstein to solve tears. If multiple tears are given
        they are solved simultaneously.

        Arguments
        ---------
            order
                List of lists of order in which to calculate nodes
            tears
                List of tear edge indexes
            iterLim
                Limit on the number of iterations to run
            tol
                Tolerance at which iteration can be stopped
            accel_min
                Minimum value for Wegstein acceleration factor
            accel_max
                Maximum value for Wegstein acceleration factor
            tol_type
                Type of tolerance value, either "abs" (absolute) or
                "rel" (relative to current value)

        Returns
        -------
            list
                List of lists of diff history, differences between input and
                output values at each iteration
        """
        hist = [] # diff at each iteration in every variable

        if not len(tears):
            # no need to iterate just run the calculations
            self.run_order(G, order, function, tears)
            return hist

        logger.info("Starting Wegstein tear convergence")

        itercount = 0
        ignore = tears + outEdges

        gofx = self.generate_gofx(G, tears)
        x = self.generate_first_x(G, tears)

        err = self.compute_err(gofx, x, tol_type)
        hist.append(err)

        if report_diffs:
            print("Diff matrix:\n%s" % err)

        # check if it's already solved
        if numpy.max(numpy.abs(err)) < tol:
            logger.info("Wegstein converged in %s iterations" % itercount)
            return hist

        # if not solved yet do one direct step
        x_prev = x
        gofx_prev = gofx
        x = gofx
        self.pass_tear_wegstein(G, tears, gofx)

        while True:
            itercount += 1

            logger.info("Running Wegstein iteration %s" % itercount)
            self.run_order(G, order, function, ignore)

            gofx = self.generate_gofx(G, tears)

            err = self.compute_err(gofx, x, tol_type)
            hist.append(err)

            if report_diffs:
                print("Diff matrix:\n%s" % err)

            if numpy.max(numpy.abs(err)) < tol:
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
            slope[numpy.isnan(slope)] = 0
            slope[numpy.isinf(slope)] = 0
            accel = slope / (slope - 1)
            accel[accel < accel_min] = accel_min
            accel[accel > accel_max] = accel_max
            x_prev = x
            gofx_prev = gofx
            x = accel * x_prev + (1 - accel) * gofx_prev
            self.pass_tear_wegstein(G, tears, x)

        self.pass_edges(G, outEdges)

        logger.info("Wegstein converged in %s iterations" % itercount)

        return hist

    def scc_collect(self, G, excludeEdges=None):
        """
        This is an algorithm for finding strongly connected components (SCCs)
        in a graph. It is based on Tarjan. 1972 Depth-First Search and Linear
        Graph Algorithms, SIAM J. Comput. v1 no. 2 1972

        Returns
        -------
            sccNodes
                List of lists of nodes in each SCC
            sccEdges
                List of lists of edge indexes in each SCC
            sccOrder
                List of lists for order in which to calculate SCCs
            outEdges
                List of lists of edge indexes leaving the SCC
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
        ndepth     = [None] * len(i2n)
        back       = [None] * len(i2n)

        # find the SCCs
        for v in range(len(i2n)):
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
        return sccNodes, sccEdges, sccOrder, outEdges

    def scc_calculation_order(self, sccNodes, ie, oe):
        """
        This determines the order in which to do calculations for strongly
        connected components. It is used to help determine the most efficient
        order to solve tear streams to prevent extra iterations. This just
        makes an adjacency list with the SCCs as nodes and calls the tree
        order function.

        Arguments
        ---------
            sccNodes
                List of lists of nodes in each SCC
            ie
                List of lists of in edge indexes to SCCs
            oe
                List of lists of out edge indexes to SCCs

        """
        adj = [] # SCC adjacency list
        adjR = [] # SCC reverse adjacency list
        # populate with empty lists before running the loop below
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
        """
        Rely on tree_order to return a calculation order of nodes

        Arguments
        ---------
            roots
                List of nodes to consider as tree roots,
                if None then the actual roots are used
            nodes
                Subset of nodes to consider in the tree,
                if None then all nodes are used
        """
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

        Arguments
        ---------
            adj
                An adjeceny list for a directed tree. This uses
                generic integer node indexes, not node names from the
                graph itself. This allows this to be used on sub-graphs
                and graps of components more easily.
            adjR
                The reverse adjacency list coresponing to adj
            roots
                List of node indexes to start from. These do not
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

        if roots is None:
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
        sccNodes, _, _, _ = self.scc_collect(G, excludeEdges=tset)
        for nodes in sccNodes:
            if len(nodes) > 1:
                return False
        return True

    def select_tear_heuristic(self, G):
        """
        This finds optimal sets of tear edges based on two criteria.
        The primary objective is to minimize the maximum number of
        times any cycle is broken. The seconday criteria is to
        minimize the number of tears.

        This function uses a branch and bound type approach.

        Returns
        -------
            tsets
                List of lists of tear sets. All the tear sets returned
                are equally good. There are often a very large number
                of equally good tear sets.
            upperbound_loop
                The max number of times any single loop is torn
            upperbound_total
                The total number of loops

        Improvemnts for the future

        I think I can imporve the efficency of this, but it is good
        enough for now. Here are some ideas for improvement:

            1. Reduce the number of redundant solutions. It is possible
            to find tears sets [1,2] and [2,1]. I eliminate
            redundent solutions from the results, but they can
            occur and it reduces efficency.

            2. Look at strongly connected components instead of whole
            graph. This would cut back on the size of graph we are
            looking at. The flowsheets are rarely one strongly
            conneted component.

            3. When you add an edge to a tear set you could reduce the
            size of the problem in the branch by only looking at
            strongly connected components with that edge removed.

            4. This returns all equally good optimal tear sets. That
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
            for edge in G.out_edges(node, keys=True):
                suc, key = edge[1], edge[2]
                if depths[suc] is None:
                    parents[suc] = node
                    cyc(suc, depth)
                elif depths[suc] < depths[node]:
                    # found a back edge, add to tear set
                    tearSet.append(edge_list.index((node, suc, key)))

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

        Returns
        -------
            edges
                List of edge indexes in the subgraph
            inEdges
                List of edge indexes starting outside the subgraph
                and ending inside
            outEdges
                List of edge indexes starting inside the subgraph
                and ending outside
        """
        e = []   # edges that connect two nodes in the subgraph
        ie = []  # in edges
        oe = []  # out edges
        edge_list = self.idx_to_edge(G)
        for i in range(G.number_of_edges()):
            src, dest, _ = edge_list[i]
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
        cycleNodes, cycleEdges = self.all_cycles(G) # call cycle finding algorithm

        # Create empty incidence matrix and then fill it out
        ceMat = numpy.zeros((len(cycleEdges), G.number_of_edges()),
                            dtype=numpy.dtype(int))
        for i in range(len(cycleEdges)):
            for e in cycleEdges[i]:
                ceMat[i, e] = 1

        return ceMat, cycleNodes, cycleEdges

    def all_cycles(self, G):
        """
        This function finds all the cycles in a directed graph.
        The algorithm is based on Tarjan 1973 Enumeration of the
        elementary circuits of a directed graph, SIAM J. Comput. v3 n2 1973.

        Returns
        -------
            cycleNodes
                List of lists of nodes in each cycle
            cycleEdges
                List of lists of edge indexes in each cycle
        """

        def backtrack(v, pre_key=None):
            # sub-function recursive part
            f = False
            pointStack.append((v, pre_key))
            mark[v] = True
            markStack.append(v)
            sucs = list(adj[v])

            for si, key in sucs:
                # iterate over successor indexes and keys
                if si < ni:
                    adj[v].remove((si, key))
                elif si == ni:
                    f = True
                    cyc = list(pointStack) # copy
                    # append the original point again so we get the last edge
                    cyc.append((si, key))
                    cycles.append(cyc)
                elif not mark[si]:
                    g = backtrack(si, key)
                    f = f or g

            if f:
                while markStack[-1] != v:
                    u = markStack.pop()
                    mark[u] = False
                markStack.pop()
                mark[v] = False

            pointStack.pop()
            return f

        i2n, adj, _ = self.adj_lists(G, multi=True)
        pointStack  = [] # stack of (node, key) tuples
        markStack = [] # nodes that have been marked
        cycles = [] # list of cycles found
        mark = [False] * len(i2n) # if a node is marked

        for ni in range(len(i2n)):
            # iterate over node indexes
            backtrack(ni)
            while len(markStack) > 0:
                i = markStack.pop()
                mark[i] = False

        # Turn node indexes back into nodes
        cycleNodes = []
        for cycle in cycles:
            cycleNodes.append([])
            for i in range(len(cycle)):
                ni, key = cycle[i]
                # change the node index in cycles to a node as well
                cycle[i] = (i2n[ni], key)
                cycleNodes[-1].append(i2n[ni])
            # pop the last node since it is the same as the first
            cycleNodes[-1].pop()

        # Now find list of edges in the cycle
        edge_map = self.edge_to_idx(G)
        cycleEdges = []
        for cyc in cycles:
            ecyc = []
            for i in range(len(cyc) - 1):
                pre, suc, key = cyc[i][0], cyc[i + 1][0], cyc[i + 1][1]
                ecyc.append(edge_map[(pre, suc, key)])
            cycleEdges.append(ecyc)

        return cycleNodes, cycleEdges

    def adj_lists(self, G, excludeEdges=None, nodes=None, multi=False):
        """
        Returns an adjacency list and a reverse adjacency list
        of node indexes for a MultiDiGraph.

        Arguments
        ---------
            G
                A networkx MultiDiGraph
            excludeEdges
                List of edge indexes to ignore when considering neighbors
            nodes
                List of nodes to form the adjacencies from
            multi
                If True, adjacency lists will contains tuples of
                (node, key) for every edge between two nodes

        Returns
        -------
            i2n
                Map from index to node for all nodes included in nodes
            adj
                Adjacency list of successor indexes
            adjR
                Reverse adjacency list of predecessor indexes
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
        i2n = [None] * len(nodes)
        n2i = dict()
        i = -1
        for node in nodes:
            i += 1
            n2i[node] = i
            i2n[i] = node

        i = -1
        for node in nodes:
            i += 1
            adj.append([])
            adjR.append([])

            seen = set()
            for edge in G.out_edges(node, keys=True):
                suc, key = edge[1], edge[2]
                if not multi and suc in seen:
                    # we only need to add the neighbor once
                    continue
                if suc in nodes and edge not in exclude:
                    # only add neighbor to seen if the edge is not excluded
                    seen.add(suc)
                    if multi:
                        adj[i].append((n2i[suc], key))
                    else:
                        adj[i].append(n2i[suc])

            seen = set()
            for edge in G.in_edges(node, keys=True):
                pre, key = edge[0], edge[2]
                if not multi and pre in seen:
                    continue
                if pre in nodes and edge not in exclude:
                    seen.add(pre)
                    if multi:
                        adjR[i].append((n2i[pre], key))
                    else:
                        adjR[i].append(n2i[pre])

        return i2n, adj, adjR
