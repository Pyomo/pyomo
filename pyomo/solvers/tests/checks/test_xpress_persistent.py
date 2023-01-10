import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.solvers.plugins.solvers.xpress_direct import xpress_available
from pyomo.opt.results.solver import TerminationCondition, SolverStatus

from pyomo.environ import value
from pyomo.opt import SolverFactory
from pyomo.common.collections import ComponentMap
import random
import math

# This serves as example as well as test case. It illustrates how to use
# callbacks with Pyomo and the Xpress solver. Of course, this is very
# solver specific since callbacks are inherently solver specific
class TSP:
    """
    Solve a MIP using cuts/constraints that are lazily separated.

    We take a random instance of the symmetric TSP and solve that using
    lazily separated constraints.

    The model is based on a graph G = (V,E).
    We have one binary variable x[e] for each edge e in E. That variable
    is set to 1 if edge e is selected in the optimal tour and 0 otherwise.

    The model contains only these explicit constraint:
        for each v in V: sum(u in V : u != v) x[uv] == 1
        for each v in V: sum(u in V : u != v) x[vu] == 1

    This states that each node must be entered and exited exactly once in the
    optimal tour.

    The above constraints ensures that the selected edges form tours. However,
    it allows multiple tours, also known as subtours. So we need a constraint
    that requires that there is only one tour (which then necessarily hits
    all the nodes). For this we just create no-good constraints for subtours:
    If S is a set of edges that forms a subtour, then we add constraint
        sum(e in S) x[e] <= |S|-1

    Since there are exponentially many subtours in a graph, this constraint
    is not stated explicitly. Instead we check for any solution that the
    optimizer finds, whether it satisfies the subtour elimination constraint.
    If it does then we accept the solution. Otherwise we reject the solution
    and augment the model by the violated subtour eliminiation constraint.

    This lazy addition of constraints is implemented using two callbacks:
    - a preintsol callback that rejects any solution that violates a
      subtour elimination constraint,
    - an optnode callback that injects any violated subtour elimination
      constraints.

    An important thing to note about this strategy is that dual reductions
    have to be disabled. Since the optimizer does not see the whole model
    (subtour elimination constraints are only generated on the fly), dual
    reductions may cut off the optimal solution.
    """
    def __init__(self, nodes, seed=0):
        """Construct a new random instance with the given seed."""
        self._nodes = nodes  # Number of nodes/cities in the instance.
        self._nodex = [0.0] * nodes # X coordinate of nodes.
        self._nodey = [0.0] * nodes # Y coordinate of nodes.

        random.seed(seed)
        for i in range(nodes):
            self._nodex[i] = 4.0 * random.random()
            self._nodey[i] = 4.0 * random.random()

    def distance(self, u, v):
        """Get the distance between two nodes."""
        return math.sqrt((self._nodex[u] - self._nodex[v]) ** 2 +
                         (self._nodey[u] - self._nodey[v]) ** 2)

    def find_tour(self, sol, prev=None):
        """Find the tour rooted at the first city in a solution.
        """
        if prev is None:
            prev = dict()
        tour = set()
        x = self._model.x

        u = 1
        used = 0
        print('1', end='')
        while True:
            for v in self._model.cities:
                if u == v:
                    continue # no self-loops
                elif v in prev:
                    continue # node already on tour
                elif sol[x[u, v]] < 0.5:
                    continue # edge not selected in solution
                elif (u, v) in tour:
                    continue # edge already on tour
                else:
                    print(' -> %d' % v, end='')
                    tour.add((u, v))
                    prev[v] = u
                    used += 1;
                    u = v;
                    break
            if u == 1:
                break
        print()
        return used

    def preintsol(self, data):
        """Integer solution check callback."""
        print("Checking feasible solution ...")

        # Get current solution and check whether it is feasible
        used = self.find_tour(data.candidate_sol)
        print("Solution is ", end='')
        if used < len(self._model.cities):
            print('infeasible (%d edges)' % used)
            data.reject = True
        else:
            print('feasible with length %f' % data.candidate_obj)

    def optnode(self, data):
        """Optimal node callback.
        This callback is invoked after the LP relaxation of a node is solved.
        This is where we can inject additional constraints as cuts.
        """
        # Only separate constraints on nodes that are integer feasible.
        if data.attributes.mipinfeas != 0:
            return

        # Get the current solution
        sol = data.getlpsol()

        # Get the tour starting at the first city and check whether it covers
        # all nodes. If it does not then it is infeasible and we must
        # generate a subtour elimination constraint.
        prev = dict()
        used = self.find_tour(sol, prev)
        if used < len(self._model.cities):
            # The tour is too short. Get the edges on the tour and add a
            # subtour elimination constraint
            lhs = sum(self._model.x[u, prev[u]] for u in self._model.cities if u in prev)
            data.addcut(lhs <= (used - 1))
            data.infeas = True

    def create_initial_tour(self):
        """Create a feasible tour and add this as initial MIP solution."""
        for u in self._model.cities:
            for v in self._model.cities:
                # A starting solution in Pyomo is set by assigning the values
                # to the variables and then passing `warmstart=True` to the
                # `solve()` call.
                if v == u + 1 or (u == self._nodes and v == 1):
                    self._model.x[u, v] = 1.0
                else:
                    self._model.x[u, v] = 0.0

    def solve(self):
        """Solve the TSP represented by this instance."""
        self._model = pe.ConcreteModel()
        self._model.cities = pe.RangeSet(self._nodes)
        # Create variables. We create one variable for each edge in
        # the complete directed graph. x[u,v] is set to 1 if the tour goes
        # from u to v, otherwise it is set to 0.
        # All variables are binary.
        self._model.x = pe.Var(self._model.cities * self._model.cities,
                               within=pe.Binary)
        self._model.cons = pe.ConstraintList()

        # Do not allow self loops.
        # We could have skipped creating the variables but fixing them to
        # 0 here is slightly easier.
        for u in self._model.cities:
            self._model.cons.add(self._model.x[u, u] <= 0.0)

        # Objective function.
        obj = sum(self._model.x[u, v] * self.distance(u-1, v-1) for u in self._model.cities for v in self._model.cities)
        self._model.obj = pe.Objective(expr=obj)

        # Constraint: Each node must be exited and entered exactly once.
        for u in self._model.cities:
            self._model.cons.add(sum(self._model.x[u, v] for v in self._model.cities if v != u) == 1)
            self._model.cons.add(sum(self._model.x[v, u] for v in self._model.cities if v != u) == 1)

        # Create a starting solution.
        # This is optional but having a feasible solution available right
        # from the beginning can improve optimizer performance.
        self.create_initial_tour()

        # We don't have all constraints explicitly in the matrix, hence
        # we must disable dual reductions. Otherwise MIP presolve may
        # cut off the optimal solution.
        opt = SolverFactory('xpress_direct')
        opt.options['mipdualreductions'] = 0

        # Add a callback that rejects solutions that do not satisfy
        # the subtour constraints.
        opt.callbacks.add_preintsol(self.preintsol)

        # Add a callback that separates subtour elimination constraints
        opt.callbacks.add_optnode(self.optnode)

        opt.solve(self._model, tee=True, warmstart=True)

        # Print the optimal tour.
        print("Tour with length %f:" % value(self._model.obj))
        self.find_tour(ComponentMap([(self._model.x[e], self._model.x[e].value) for e in self._model.x]))

        self._model = None # cleanup


class TestXpressPersistent(unittest.TestCase):
    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_basics(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-10, 10))
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y >= 2*m.x + 1)

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)

        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        res = opt.solve()
        self.assertAlmostEqual(m.x.value, -0.4, delta=1e-6)
        self.assertAlmostEqual(m.y.value, 0.2, delta=1e-6)
        opt.load_duals()
        self.assertAlmostEqual(m.dual[m.c1], -0.4, delta=1e-6)
        del m.dual

        m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 2)

        res = opt.solve(save_results=False, load_solutions=False)
        self.assertAlmostEqual(m.x.value, -0.4, delta=1e-6)
        self.assertAlmostEqual(m.y.value, 0.2, delta=1e-6)
        opt.load_vars()
        self.assertAlmostEqual(m.x.value, 0, delta=1e-6)
        self.assertAlmostEqual(m.y.value, 1, delta=2e-6)

        opt.remove_constraint(m.c2)
        m.del_component(m.c2)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        self.assertEqual(opt.get_xpress_control('feastol'), 1e-6)
        res = opt.solve(options={'feastol': '1e-7'})
        self.assertEqual(opt.get_xpress_control('feastol'), 1e-7)
        self.assertAlmostEqual(m.x.value, -0.4, delta=1e-6)
        self.assertAlmostEqual(m.y.value, 0.2, delta=1e-6)

        m.x.setlb(-5)
        m.x.setub(5)
        opt.update_var(m.x)
        # a nice wrapper for xpress isn't implemented,
        # so we'll do this directly
        x_idx = opt._solver_model.getIndex(opt._pyomo_var_to_solver_var_map[m.x])
        lb = []
        opt._solver_model.getlb(lb, x_idx, x_idx)
        ub = []
        opt._solver_model.getub(ub, x_idx, x_idx)
        self.assertEqual(lb[0], -5)
        self.assertEqual(ub[0], 5)

        m.x.fix(0)
        opt.update_var(m.x)
        lb = []
        opt._solver_model.getlb(lb, x_idx, x_idx)
        ub = []
        opt._solver_model.getub(ub, x_idx, x_idx)
        self.assertEqual(lb[0], 0)
        self.assertEqual(ub[0], 0)

        m.x.unfix()
        opt.update_var(m.x)
        lb = []
        opt._solver_model.getlb(lb, x_idx, x_idx)
        ub = []
        opt._solver_model.getub(ub, x_idx, x_idx)
        self.assertEqual(lb[0], -5)
        self.assertEqual(ub[0], 5)

        m.c2 = pe.Constraint(expr=m.y >= m.x**2)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 2)

        opt.remove_constraint(m.c2)
        m.del_component(m.c2)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        m.z = pe.Var()
        opt.add_var(m.z)
        self.assertEqual(opt.get_xpress_attribute('cols'), 3)
        opt.remove_var(m.z)
        del m.z
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_qconstraint(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x**2 + m.y**2)

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        opt.remove_constraint(m.c1)
        self.assertEqual(opt.get_xpress_attribute('rows'), 0)

        opt.add_constraint(m.c1)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_lconstraint(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c2 = pe.Constraint(expr=m.x + m.y == 1)

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        opt.remove_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 0)

        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_sosconstraint(self):
        m = pe.ConcreteModel()
        m.a = pe.Set(initialize=[1,2,3], ordered=True)
        m.x = pe.Var(m.a, within=pe.Binary)
        m.y = pe.Var(within=pe.Binary)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.SOSConstraint(var=m.x, sos=1)

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('sets'), 1)

        opt.remove_sos_constraint(m.c1)
        self.assertEqual(opt.get_xpress_attribute('sets'), 0)

        opt.add_sos_constraint(m.c1)
        self.assertEqual(opt.get_xpress_attribute('sets'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_sosconstraint2(self):
        m = pe.ConcreteModel()
        m.a = pe.Set(initialize=[1,2,3], ordered=True)
        m.x = pe.Var(m.a, within=pe.Binary)
        m.y = pe.Var(within=pe.Binary)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.SOSConstraint(var=m.x, sos=1)

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('sets'), 1)
        m.c2 = pe.SOSConstraint(var=m.x, sos=2)
        opt.add_sos_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('sets'), 2)
        opt.remove_sos_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('sets'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_var(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)

        opt.remove_var(m.x)
        self.assertEqual(opt.get_xpress_attribute('cols'), 1)

        opt.add_var(m.x)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)

        opt.remove_var(m.x)
        opt.add_var(m.x)
        opt.remove_var(m.x)
        self.assertEqual(opt.get_xpress_attribute('cols'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_column(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(within=pe.NonNegativeReals)
        m.c = pe.Constraint(expr=(0, m.x, 1))
        m.obj = pe.Objective(expr=-m.x)

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        opt.solve()
        self.assertAlmostEqual(m.x.value, 1)

        m.y = pe.Var(within=pe.NonNegativeReals)

        opt.add_column(m, m.y, -3, [m.c], [2])
        opt.solve()

        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 0.5)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_column_exceptions(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.c = pe.Constraint(expr=(0, m.x, 1))
        m.ci = pe.Constraint([1,2], rule=lambda m,i:(0,m.x,i+1))
        m.cd = pe.Constraint(expr=(0, -m.x, 1))
        m.cd.deactivate()
        m.obj = pe.Objective(expr=-m.x)

        opt = pe.SolverFactory('xpress_persistent')

        # set_instance not called
        self.assertRaises(RuntimeError, opt.add_column, m, m.x, 0, [m.c], [1])

        opt.set_instance(m)

        m2 = pe.ConcreteModel()
        m2.y = pe.Var()
        m2.c = pe.Constraint(expr=(0,m.x,1))

        # different model than attached to opt
        self.assertRaises(RuntimeError, opt.add_column, m2, m2.y, 0, [], [])
        # pyomo var attached to different model
        self.assertRaises(RuntimeError, opt.add_column, m, m2.y, 0, [], [])

        z = pe.Var()
        # pyomo var floating
        self.assertRaises(RuntimeError, opt.add_column, m, z, -2, [m.c, z], [1])

        m.y = pe.Var()
        # len(coefficents) == len(constraints)
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c], [1,2])
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c, z], [1])

        # add indexed constraint
        self.assertRaises(AttributeError, opt.add_column, m, m.y, -2, [m.ci], [1])
        # add something not a _ConstraintData
        self.assertRaises(AttributeError, opt.add_column, m, m.y, -2, [m.x], [1])

        # constraint not on solver model
        self.assertRaises(KeyError, opt.add_column, m, m.y, -2, [m2.c], [1])

        # inactive constraint
        self.assertRaises(KeyError, opt.add_column, m, m.y, -2, [m.cd], [1])

        opt.add_var(m.y)
        # var already in solver model
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c], [1])

    def _markshare(self):
        """Create a model that is non-trivial to solve.
        The returned model has two variables: `x` and `s`. It also has an
        objective function that is stored in `obj`.
        """
        model = pe.ConcreteModel()
        model.X = pe.RangeSet(50)
        model.S = pe.RangeSet(6)
        model.x = pe.Var(model.X, within=pe.Binary)
        x = model.x
        model.s = pe.Var(model.S, bounds = (0, None))
        s = model.s
        model.obj = pe.Objective(expr=s[1] + s[2] + s[3] + s[4] + s[5] + s[6])
        model.cons = pe.ConstraintList()
        
        model.cons.add(s[1] + 25*x[1] + 35*x[2] + 14*x[3] + 76*x[4] + 58*x[5] + 10*x[6] + 20*x[7]
                       + 51*x[8] + 58*x[9] + x[10] + 35*x[11] + 40*x[12] + 65*x[13] + 59*x[14] + 24*x[15]
                       + 44*x[16] + x[17] + 93*x[18] + 24*x[19] + 68*x[20] + 38*x[21] + 64*x[22] + 93*x[23]
                       + 14*x[24] + 83*x[25] + 6*x[26] + 58*x[27] + 14*x[28] + 71*x[29] + 17*x[30]
                       + 18*x[31] + 8*x[32] + 57*x[33] + 48*x[34] + 35*x[35] + 13*x[36] + 47*x[37]
                       + 46*x[38] + 8*x[39] + 82*x[40] + 51*x[41] + 49*x[42] + 85*x[43] + 66*x[44]
                       + 45*x[45] + 99*x[46] + 21*x[47] + 75*x[48] + 78*x[49] + 43*x[50] == 1116)
        model.cons.add(s[2] + 97*x[1] + 64*x[2] + 24*x[3] + 63*x[4] + 58*x[5] + 45*x[6] + 20*x[7]
                       + 71*x[8] + 32*x[9] + 7*x[10] + 28*x[11] + 77*x[12] + 95*x[13] + 96*x[14]
                       + 70*x[15] + 22*x[16] + 93*x[17] + 32*x[18] + 17*x[19] + 56*x[20] + 74*x[21]
                       + 62*x[22] + 94*x[23] + 9*x[24] + 92*x[25] + 90*x[26] + 40*x[27] + 45*x[28]
                       + 84*x[29] + 62*x[30] + 62*x[31] + 34*x[32] + 21*x[33] + 2*x[34] + 75*x[35]
                       + 42*x[36] + 75*x[37] + 29*x[38] + 4*x[39] + 64*x[40] + 80*x[41] + 17*x[42]
                       + 55*x[43] + 73*x[44] + 23*x[45] + 13*x[46] + 91*x[47] + 70*x[48] + 73*x[49]
                       + 28*x[50] == 1325)
        model.cons.add(s[3] + 95*x[1] + 71*x[2] + 19*x[3] + 15*x[4] + 66*x[5] + 76*x[6] + 4*x[7]
                       + 50*x[8] + 50*x[9] + 97*x[10] + 83*x[11] + 14*x[12] + 27*x[13] + 14*x[14]
                       + 34*x[15] + 9*x[16] + 99*x[17] + 62*x[18] + 92*x[19] + 39*x[20] + 56*x[21]
                       + 53*x[22] + 91*x[23] + 81*x[24] + 46*x[25] + 94*x[26] + 76*x[27] + 53*x[28]
                       + 58*x[29] + 23*x[30] + 15*x[31] + 63*x[32] + 2*x[33] + 31*x[34] + 55*x[35]
                       + 71*x[36] + 97*x[37] + 71*x[38] + 55*x[39] + 8*x[40] + 57*x[41] + 14*x[42]
                       + 76*x[43] + x[44] + 46*x[45] + 87*x[46] + 22*x[47] + 97*x[48] + 99*x[49] + 92*x[50]
                       == 1353)
        model.cons.add(s[4] + x[1] + 27*x[2] + 46*x[3] + 48*x[4] + 66*x[5] + 58*x[6] + 52*x[7] + 6*x[8]
                       + 14*x[9] + 26*x[10] + 55*x[11] + 61*x[12] + 60*x[13] + 3*x[14] + 33*x[15]
                       + 99*x[16] + 36*x[17] + 55*x[18] + 70*x[19] + 73*x[20] + 70*x[21] + 38*x[22]
                       + 66*x[23] + 39*x[24] + 43*x[25] + 63*x[26] + 88*x[27] + 47*x[28] + 18*x[29]
                       + 73*x[30] + 40*x[31] + 91*x[32] + 96*x[33] + 49*x[34] + 13*x[35] + 27*x[36]
                       + 22*x[37] + 71*x[38] + 99*x[39] + 66*x[40] + 57*x[41] + x[42] + 54*x[43] + 35*x[44]
                       + 52*x[45] + 66*x[46] + 26*x[47] + x[48] + 26*x[49] + 12*x[50] == 1169)
        model.cons.add(s[5] + 3*x[1] + 94*x[2] + 51*x[3] + 4*x[4] + 25*x[5] + 46*x[6] + 30*x[7]
                       + 2*x[8] + 89*x[9] + 65*x[10] + 28*x[11] + 46*x[12] + 36*x[13] + 53*x[14]
                       + 30*x[15] + 73*x[16] + 37*x[17] + 60*x[18] + 21*x[19] + 41*x[20] + 2*x[21]
                       + 21*x[22] + 93*x[23] + 82*x[24] + 16*x[25] + 97*x[26] + 75*x[27] + 50*x[28]
                       + 13*x[29] + 43*x[30] + 45*x[31] + 64*x[32] + 78*x[33] + 78*x[34] + 6*x[35]
                       + 35*x[36] + 72*x[37] + 31*x[38] + 28*x[39] + 56*x[40] + 60*x[41] + 23*x[42]
                       + 70*x[43] + 46*x[44] + 88*x[45] + 20*x[46] + 69*x[47] + 13*x[48] + 40*x[49]
                       + 73*x[50] == 1160)
        model.cons.add(s[6] + 69*x[1] + 72*x[2] + 94*x[3] + 56*x[4] + 90*x[5] + 20*x[6] + 56*x[7]
                       + 50*x[8] + 79*x[9] + 59*x[10] + 36*x[11] + 24*x[12] + 42*x[13] + 9*x[14]
                       + 29*x[15] + 68*x[16] + 10*x[17] + x[18] + 44*x[19] + 74*x[20] + 61*x[21] + 37*x[22]
                       + 71*x[23] + 63*x[24] + 44*x[25] + 77*x[26] + 57*x[27] + 46*x[28] + 51*x[29]
                       + 43*x[30] + 4*x[31] + 85*x[32] + 59*x[33] + 7*x[34] + 25*x[35] + 46*x[36] + 25*x[37]
                       + 70*x[38] + 78*x[39] + 88*x[40] + 20*x[41] + 40*x[42] + 40*x[43] + 16*x[44]
                       + 3*x[45] + 3*x[46] + 5*x[47] + 77*x[48] + 88*x[49] + 16*x[50] == 1163)
        
        return model
        

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_callbacks_01(self):
        """Simple callback test.

        Tests that optnode, preintsol, intsol callbacks are invoked.
        Also tests that information between preintsol an intsol callbacks
        is consistent.
        """
        model = self._markshare()
        opt = pe.SolverFactory('xpress_direct')
        opt.options['MAXNODE'] = 5
        opt.options['THREADS'] = 1 # for interaction between preintsol and intsol
        test = self

        lastnode = [0]
        noptnode = [0]
        def optnode(data):
            try:
                noptnode[0] += 1
                node = data.attributes.nodes
                test.assertGreaterEqual(node, lastnode[0])
                lastnode[0] = node
            except Exception as ex:
                print('optnode:', ex)
                raise ex
        opt.callbacks.add_optnode(optnode)

        announced = [None]
        npreintsol = [0]
        def preintsol(data):
            try:
                npreintsol[0] += 1
                test.assertIsNone(announced[0])
                test.assertGreater(data.attributes.mipobjval,
                                   data.candidate_obj)
                announced[0] = (data.candidate_obj, data.candidate_sol)
            except Exception as ex:
                print('preintsol:', ex)
                raise ex
        opt.callbacks.add_preintsol(preintsol)

        nintsol = [0]
        def intsol(data):
            try:
                nintsol[0] += 1
                self.assertIsNotNone(announced[0])
                obj, x = announced[0]
                announced[0] = None
                self.assertEqual(data.objval, obj)
                sol = data.solution
                for p in sol:
                    self.assertEqual(sol[p], x[p])
            except Exception as ex:
                print('intsol:', ex)
                raise ex
        opt.callbacks.add_intsol(intsol)

        opt.solve(model, tee=True)

        self.assertGreater(noptnode[0], 0)
        self.assertGreater(npreintsol[0], 0)
        self.assertGreater(nintsol[0], 0)
        self.assertGreaterEqual(lastnode[0], opt.options['MAXNODE'])

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_callbacks_02(self):
        """Test branching callback by doing most fractional branching on markshare."""

        model = self._markshare()
        opt = pe.SolverFactory('xpress_direct')
        opt.options['MAXNODE'] = 5
        test = self
        
        called = [0]
        def chgbranchobject(data):
            try:
                test.assertIsNotNone(data.branchobject)
                test.assertEqual(data.branchobject, data.orig_branchobject)
                called[0] += 1
                sol = data.getlpsol()
                maxfrac = 0.0
                maxvar = None
                for var in sol:
                    if var.domain == pe.Binary:
                        frac = abs(round(sol[var]) - sol.var)
                        if frac > maxfrac:
                            maxfrac = frac
                            maxvar = var
                test.assertIsNotNone(maxvar)
                if maxvar is not None:
                    b = data.new_object(2)
                    b.addbounds(0, ['U'], [data.map_pyomo_var(maxvar)], [0])
                    b.addbounds(1, ['L'], [data.map_pyomo_var(maxvar)], [1])
                    data.branchobject = b
            except Exception as ex:
                print('chgbranchobject:', ex)
                raise ex
        opt.callbacks.add_chgbranchobject(chgbranchobject)

        opt.solve(model, tee=True)

        self.assertGreater(called[0], 0)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_callbacks_03(self):
        """Test the TSP example."""
        TSP(10).solve()

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_nonconvexqp_locally_optimal(self):
        """Test non-convex QP for which xpress_direct should find a locally
        optimal solution."""
        m = pe.ConcreteModel()
        m.x1 = pe.Var()
        m.x2 = pe.Var()
        m.x3 = pe.Var()

        m.obj = pe.Objective(rule=lambda m: 2 * m.x1 + m.x2 + m.x3,
                             sense=pe.minimize)
        m.equ1 = pe.Constraint(rule=lambda m: m.x1 + m.x2 + m.x3 == 1)
        m.cone = pe.Constraint(rule=lambda m: m.x2 * m.x2 + m.x3 * m.x3 <= m.x1 * m.x1)
        m.equ2 = pe.Constraint(rule=lambda m: m.x1 >= 0)

        opt = pe.SolverFactory('xpress_direct')
        opt.options['XSLP_SOLVER'] = 0

        results = opt.solve(m)
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.locallyOptimal)

        # Cannot test exact values since the may be different depending on
        # random effects. So just test all are non-zero.
        self.assertGreater(m.x1.value, 0.0)
        self.assertGreater(m.x2.value, 0.0)
        self.assertGreater(m.x3.value, 0.0)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_nonconvexqp_infeasible(self):
        """Test non-convex QP which xpress_direct should prove infeasible."""
        m = pe.ConcreteModel()
        m.x1 = pe.Var()
        m.x2 = pe.Var()
        m.x3 = pe.Var()

        m.obj = pe.Objective(rule=lambda m: 2 * m.x1 + m.x2 + m.x3,
                             sense=pe.minimize)
        m.equ1a = pe.Constraint(rule=lambda m: m.x1 + m.x2 + m.x3 == 1)
        m.equ1b = pe.Constraint(rule=lambda m: m.x1 + m.x2 + m.x3 == -1)
        m.cone = pe.Constraint(rule=lambda m: m.x2 * m.x2 + m.x3 * m.x3 <= m.x1 * m.x1)
        m.equ2 = pe.Constraint(rule=lambda m: m.x1 >= 0)

        opt = pe.SolverFactory('xpress_direct')
        opt.options['XSLP_SOLVER'] = 0

        results = opt.solve(m)
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)
