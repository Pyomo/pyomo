\* Source Pyomo model name=unknown *\

max 
sub_dual_o:
+5 sub_dual_c1
+3 sub_dual_c2
+4 sub_dual_c3

s.t.

c_u_sub_dual_x1_:
+4 sub_dual_c1
+1 sub_dual_c2
-6 z
<= 0

c_u_sub_dual_x2_:
+2 sub_dual_c1
+1 sub_dual_c2
+1 sub_dual_c3
-4 z
<= 0

c_u_sub_dual_x3_:
+1 sub_dual_c1
+1 sub_dual_c3
-2 z
<= 0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
    -inf <= z <= 0
   0 <= sub_dual_c1 <= +inf
   0 <= sub_dual_c2 <= +inf
   0 <= sub_dual_c3 <= +inf
end
