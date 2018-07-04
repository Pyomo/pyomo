\* Source Pyomo model name=unknown *\

min 
x7:
+1 x6

s.t.

r_l_x8_:
+0 ONE_VAR_CONSTANT
>= -2

r_u_x8_:
+0 ONE_VAR_CONSTANT
<= 0

r_l_x9_:
+1 x2
>= -1

r_u_x9_:
+1 x2
<= 1

r_l_x10_:
+1 x3
>= -1

r_u_x10_:
+1 x3
<= 1

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
    -inf <= x2 <= +inf
    -inf <= x3 <= +inf
    -inf <= x6 <= +inf
end
