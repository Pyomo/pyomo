min 
objective:
-6.000000 x(1)
-6.000000 x(2)
-6.000000 x(3)
+ [
+ 2.0 x(1) ^ 2
+ 2.0 x(2) ^ 2
+ 2.0 x(3) ^ 2
] / 2
+32.000000 ONE_VAR_CONSTANT

s.t.

r_l_bound_x(1)_:
+1.000000 x(1)
>= -10.0

r_u_bound_x(1)_:
+1.000000 x(1)
<= 10.0

r_l_bound_x(2)_:
+1.000000 x(2)
>= -10.0

r_u_bound_x(2)_:
+1.000000 x(2)
<= 10.0

r_l_bound_x(3)_:
+1.000000 x(3)
>= -10.0

r_u_bound_x(3)_:
+1.000000 x(3)
<= 10.0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
    -inf <= x(1) <= +inf
    -inf <= x(2) <= +inf
    -inf <= x(3) <= +inf
end
