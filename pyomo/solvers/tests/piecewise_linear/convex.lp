\* Source Pyomo model name=unknown *\

min 
obj:
+1 Z

s.t.

c_u_con_simplified_piecewise_constraint(1)_:
-1 Z
-1.0 X
<= -2.0

c_u_con_simplified_piecewise_constraint(2)_:
-1 Z
+1 X
<= 0.0

bounds
   -inf <= Z <= +inf
   -5 <= X <= 5
end
