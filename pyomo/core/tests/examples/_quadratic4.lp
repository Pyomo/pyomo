min 
objective:
+1.000000 x
+1.000000 y
+ [
+ 1.0 x ^ 2
+ 7.0 y ^ 2
+ 4.0 x * y
] / 2

s.t.

c_l_constraint_:
+1.000000 x
+1.000000 y
>= 10.0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds 
   0 <= x <= +inf
   0 <= y <= +inf
end 
