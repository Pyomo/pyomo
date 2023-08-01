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
from math import ceil, log2

@TransformationFactory.register(
    'contrib.piecewise.disaggregated_logarithmic',
    doc="TODO document",
)
class NestedInnerRepresentationGDPTransformation(PiecewiseLinearToGDP):
    """
    Represent a piecewise linear function "logarithmically" by using a MIP with
    log_2(|P|) binary decision variables. This method of logarithmically 
    formulating the piecewise linear function imposes no restrictions on the 
    family of polytopes. This method is due to Vielma et al., 2010.
    """
    CONFIG = PiecewiseLinearToGDP.CONFIG()
    _transformation_name = 'pw_linear_disaggregated_log'
     
    # Implement to use PiecewiseLinearToGDP. This function returns the Var
    # that replaces the transformed piecewise linear expr
    def _transform_pw_linear_expr(self, pw_expr, pw_linear_func, transformation_block):
        self.DEBUG = False
        # Get a new Block() in transformation_block.transformed_functions, which
        # is a Block(Any)
        transBlock = transformation_block.transformed_functions[
            len(transformation_block.transformed_functions)
        ]

        dimension = pw_expr.nargs()
        substitute_var = transBlock.substitute_var = Var()
        pw_linear_func.map_transformation_var(pw_expr, substitute_var)
        self.substitute_var_lb = float('inf')
        self.substitute_var_ub = -float('inf')

        simplices = pw_linear_func._simplices
        num_simplices = len(simplices)
        simplex_indices = range(num_simplices)
        # Assumption: the simplices are really simplices and all have the same number of points
        simplex_point_indices = range(len(simplices[0]))

        choices = list(zip(pw_linear_func._simplices, pw_linear_func._linear_functions))

        log_dimension = ceil(log2(num_simplices))
        binaries = transBlock.binaries = Var(range(log_dimension), domain=Binary)

        # injective function \mathcal{P} -> ceil(log_2(|P|)) used to identify simplices
        # (really just polytopes are required) with binary vectors
        B = {}
        for i, p in enumerate(simplices):
            B[id(p)] = self._get_binary_vector(i, log_dimension)
        
        # The lambdas \lambda_{P,v}
        lambdas = transBlock.lambdas = Var(simplex_indices, simplex_point_indices, bounds=(0, 1))
        transBlock.convex_combo = Constraint(sum(lambdas[P, v] for P in simplex_indices for v in simplex_point_indices) == 1)

        # The branching rules, establishing using the binaries that only one simplex's lambdas
        # may be nonzero
        @transBlock.Constraint(range(log_dimension))
        def simplex_choice_1(b, l):
            return (
                sum(lambdas[P, v] for P in self._P_plus(B, l) for v in simplex_point_indices) <= binaries[l]
            )
        @transBlock.Constraint(range(log_dimension))
        def simplex_choice_2(b, l):
            return (
                sum(lambdas[P, v] for P in self._P_0(B, l) for v in simplex_point_indices) <= 1 - binaries[l]
            )
        
        #for i, (simplex, pwlf) in enumerate(choices):
        # x_i = sum(lambda_P,v v_i)
        @transBlock.Constraint(range(dimension))
        def x_constraint(b, i):
            return sum([stuff] for )


        #linear_func_expr = linear_func(*pw_expr.args)
        ## Make the substitute Var equal the PWLE
        #b.set_substitute = Constraint(expr=root_block.substitute_var == linear_func_expr)
    
    # Not a gray code, just a regular binary representation
    # TODO this is probably not optimal, test the gray codes too
    def _get_binary_vector(self, num, length):
        if ceil(log2(num)) > length:
            raise DeveloperError("Invalid input in _get_binary_vector")
        # Use python's string formatting instead of bothering with modular
        # arithmetic. May be slow.
        return (int(x) for x in format(num, f'0{length}b'))

    # Return {P \in \mathcal{P} | B(P)_l = 0}
    def _P_0(B, l, simplices):
        return [p for p in simplices if B[id(p)][l] == 0]
    # Return {P \in \mathcal{P} | B(P)_l = 1}
    def _P_plus(B, l, simplices):
        return [p for p in simplices if B[id(p)][l] == 1]