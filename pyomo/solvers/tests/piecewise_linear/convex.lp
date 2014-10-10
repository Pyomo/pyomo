\* Source Pyomo model name=unknown *\

min 
obj:
+1 Z

s.t.

c_u_con_simplified_piecewise_constraint(1)_:
-1 X
-1 Z
<= -2

c_u_con_simplified_piecewise_constraint(2)_:
+1 X
-1 Z
<= 0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds 
   -5 <= X <= 5
    -inf <= Z <= +inf
end 
