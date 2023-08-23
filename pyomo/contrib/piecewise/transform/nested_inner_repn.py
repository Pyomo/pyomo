from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.piecewise.transform.piecewise_to_gdp_transformation import (
    PiecewiseLinearToGDP,
)
from pyomo.core import Constraint, Binary, NonNegativeIntegers, Suffix, Var
from pyomo.core.base import TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
from pyomo.common.errors import DeveloperError
from pyomo.core.expr.visitor import SimpleExpressionVisitor
from pyomo.core.expr.current import identify_components

@TransformationFactory.register(
    'contrib.piecewise.nested_inner_repn_gdp',
    doc="TODO document",
)
class NestedInnerRepresentationGDPTransformation(PiecewiseLinearToGDP):
    """
    Represent a piecewise linear function "logarithmically" by using a nested
    GDP to determine which polytope a point is in, then representing it as a
    convex combination of extreme points, with multipliers "local" to that 
    particular polytope, i.e., not shared with neighbors. This method of 
    logarithmically formulating the piecewise linear function imposes no 
    restrictions on the family of polytopes. We rely on the identification of 
    variables to make this logarithmic in the number of binaries. This method 
    is due to Vielma et al., 2010.
    """
    CONFIG = PiecewiseLinearToGDP.CONFIG()
    _transformation_name = 'pw_linear_nested_inner_repn'
     
    # Implement to use PiecewiseLinearToGDP. This function returns the Var
    # that replaces the transformed piecewise linear expr
    def _transform_pw_linear_expr(self, pw_expr, pw_linear_func, transformation_block):
        self.DEBUG = False
        identify_vars = False
        # Get a new Block() in transformation_block.transformed_functions, which
        # is a Block(Any)
        transBlock = transformation_block.transformed_functions[
            len(transformation_block.transformed_functions)
        ]

        # these copy-pasted lines (from inner_representation_gdp) seem useful
        # adding some of this stuff to self so I don't have to pass it around
        self.pw_linear_func = pw_linear_func
        # map number -> list of Disjuncts which contain Disjunctions at that level
        self.disjunct_levels = {}
        self.dimension = pw_expr.nargs()
        substitute_var = transBlock.substitute_var = Var()
        pw_linear_func.map_transformation_var(pw_expr, substitute_var)
        self.substitute_var_lb = float('inf')
        self.substitute_var_ub = -float('inf')
        
        choices = list(zip(pw_linear_func._simplices, pw_linear_func._linear_functions))

        if self.DEBUG:
            print(f"dimension is {self.dimension}")

        # If there was only one choice, don't bother making a disjunction, just
        # use the linear function directly (but still use the substitute_var for 
        # consistency).
        if len(choices) == 1:
            (_, linear_func) = choices[0] # simplex isn't important in this case
            linear_func_expr = linear_func(*pw_expr.args)
            transBlock.set_substitute = Constraint(expr=substitute_var == linear_func_expr)
            (self.substitute_var_lb, self.substitute_var_ub) = compute_bounds_on_expr(linear_func_expr)
        else:
            # Add the disjunction
            transBlock.disj = self._get_disjunction(choices, transBlock, pw_expr, transBlock, 1)

        # Widen bounds as determined when setting up the disjunction
        if self.substitute_var_lb < float('inf'):
            transBlock.substitute_var.setlb(self.substitute_var_lb)
        if self.substitute_var_ub > -float('inf'):
            transBlock.substitute_var.setub(self.substitute_var_ub)
        
        if self.DEBUG:
            print(f"lb is {self.substitute_var_lb}, ub is {self.substitute_var_ub}")
        
        # NOTE - This functionality does not work. Even when we can choose the indicator
        # variables, it seems that infeasibilities will always be generated. We may need
        # to just directly transform to mip :(
        if identify_vars:
            if self.DEBUG:
                print("Now identifying variables")
                for i in self.disjunct_levels.keys():
                    print(f"level {i}: {len(self.disjunct_levels[i])} disjuncts")
            transBlock.var_identifications_l = Constraint(NonNegativeIntegers, NonNegativeIntegers)
            transBlock.var_identifications_r = Constraint(NonNegativeIntegers, NonNegativeIntegers)
            for k in self.disjunct_levels.keys():
                disj_0 = self.disjunct_levels[k][0]
                for i, disj in enumerate(self.disjunct_levels[k][1:]):
                    transBlock.var_identifications_l[k, i] = disj.d_l.binary_indicator_var == disj_0.d_l.binary_indicator_var
                    transBlock.var_identifications_r[k, i] = disj.d_r.binary_indicator_var == disj_0.d_r.binary_indicator_var
        return substitute_var

    # Recursively form the Disjunctions and Disjuncts. This shouldn't blow up
    # the stack, since the whole point is that we'll only go logarithmically
    # many calls deep.
    def _get_disjunction(self, choices, parent_block, pw_expr, root_block, level):
        size = len(choices)
        if self.DEBUG:
            print(f"calling _get_disjunction with size={size}")
        # Our base cases will be 3 and 2, since it would be silly to construct
        # a Disjunction containing only one Disjunct. We can ensure that size
        # is never 1 unless it was only passsed a single choice from the start,
        # which we can handle before calling.
        if size > 3: 
            half = size // 2 # (integer divide)
            # This tree will be slightly heavier on the right side
            choices_l = choices[:half]
            choices_r = choices[half:]
            # Is this valid Pyomo?
            @parent_block.Disjunct()
            def d_l(b):
                b.inner_disjunction_l = self._get_disjunction(choices_l, b, pw_expr, root_block, level + 1)
            @parent_block.Disjunct()
            def d_r(b):
                b.inner_disjunction_r = self._get_disjunction(choices_r, b, pw_expr, root_block, level + 1)
            if level not in self.disjunct_levels.keys():
                self.disjunct_levels[level] = []
            self.disjunct_levels[level].append(parent_block.d_l)
            self.disjunct_levels[level].append(parent_block.d_r)
            return Disjunction(expr=[parent_block.d_l, parent_block.d_r])
        elif size == 3:
            # Let's stay heavier on the right side for consistency. So the left
            # Disjunct will be the one to contain constraints, rather than a
            # Disjunction
            @parent_block.Disjunct()
            def d_l(b):
                simplex, linear_func = choices[0]
                self._set_disjunct_block_constraints(b, simplex, linear_func, pw_expr, root_block)
            @parent_block.Disjunct()
            def d_r(b):
                b.inner_disjunction_r = self._get_disjunction(choices[1:], b, pw_expr, root_block, level + 1)
            if level not in self.disjunct_levels.keys():
                self.disjunct_levels[level] = []
            self.disjunct_levels[level].append(parent_block.d_r)
            return Disjunction(expr=[parent_block.d_l, parent_block.d_r])
        elif size == 2:
            # In this case both sides are regular Disjuncts
            @parent_block.Disjunct()
            def d_l(b):
                simplex, linear_func = choices[0]
                self._set_disjunct_block_constraints(b, simplex, linear_func, pw_expr, root_block)
            @parent_block.Disjunct()
            def d_r(b):
                simplex, linear_func = choices[1]
                self._set_disjunct_block_constraints(b, simplex, linear_func, pw_expr, root_block)
            return Disjunction(expr=[parent_block.d_l, parent_block.d_r])
        else:
            raise DeveloperError("Unreachable: 1 or 0 choices were passed to "
                                 "_get_disjunction in nested_inner_repn.py.")

    def _set_disjunct_block_constraints(self, b, simplex, linear_func, pw_expr, root_block):
        # Define the lambdas sparsely like in the version I'm copying,
        # only the first few will participate in constraints
        b.lambdas = Var(NonNegativeIntegers, dense=False, bounds=(0, 1))
        # Get the extreme points to add up
        extreme_pts = []
        for idx in simplex:
            extreme_pts.append(self.pw_linear_func._points[idx])
        # Constrain sum(lambda_i) = 1     
        b.convex_combo = Constraint(
            expr=sum(b.lambdas[i] for i in range(len(extreme_pts))) == 1
        )
        linear_func_expr = linear_func(*pw_expr.args)
        # Make the substitute Var equal the PWLE
        b.set_substitute = Constraint(expr=root_block.substitute_var == linear_func_expr)
        # Widen the variable bounds to those of this linear func expression
        (lb, ub) = compute_bounds_on_expr(linear_func_expr)
        if lb is not None and lb < self.substitute_var_lb:
            self.substitute_var_lb = lb
        if ub is not None and ub > self.substitute_var_ub:
            self.substitute_var_ub = ub
        # Constrain x = \sum \lambda_i v_i
        @b.Constraint(range(self.dimension))
        def linear_combo(d, i):
            return pw_expr.args[i] == sum(
                d.lambdas[j] * pt[i] for j, pt in enumerate(extreme_pts)
            )
        # Mark the lambdas as local in order to prevent disagreggating multiple
        # times in the hull transformation
        b.LocalVars = Suffix(direction=Suffix.LOCAL)
        b.LocalVars[b] = [v for v in b.lambdas.values()]
