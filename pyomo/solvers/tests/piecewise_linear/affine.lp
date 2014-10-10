\* Source Pyomo model name=unknown *\

min 
obj:
+1 n
+1 p

s.t.

c_e_pn_con_:
+1 Z
+1 n
-1 p
= 7

c_e_con_DCC_constraint1_:
+1 X
+1 con_DCC_lambda(1_1)
-2 con_DCC_lambda(1_2)
-2 con_DCC_lambda(2_2)
-6 con_DCC_lambda(2_3)
-6 con_DCC_lambda(3_3)
-10 con_DCC_lambda(3_4)
= 0

c_e_con_DCC_constraint2_:
+1 Z
+1 con_DCC_lambda(1_1)
+8 con_DCC_lambda(2_3)
+8 con_DCC_lambda(3_3)
-12 con_DCC_lambda(3_4)
= 0

c_e_con_DCC_constraint3(1)_:
+1 con_DCC_bin_y(1)
-1 con_DCC_lambda(1_1)
-1 con_DCC_lambda(1_2)
= 0

c_e_con_DCC_constraint3(2)_:
+1 con_DCC_bin_y(2)
-1 con_DCC_lambda(2_2)
-1 con_DCC_lambda(2_3)
= 0

c_e_con_DCC_constraint3(3)_:
+1 con_DCC_bin_y(3)
-1 con_DCC_lambda(3_3)
-1 con_DCC_lambda(3_4)
= 0

c_e_con_DCC_constraint4_:
+1 con_DCC_bin_y(1)
+1 con_DCC_bin_y(2)
+1 con_DCC_bin_y(3)
= 1

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds 
   -1 <= X <= 10
    -inf <= Z <= +inf
   0 <= p <= +inf
   0 <= n <= +inf
   0 <= con_DCC_lambda(1_1) <= +inf
   0 <= con_DCC_lambda(1_2) <= +inf
   0 <= con_DCC_lambda(2_2) <= +inf
   0 <= con_DCC_lambda(2_3) <= +inf
   0 <= con_DCC_lambda(3_3) <= +inf
   0 <= con_DCC_lambda(3_4) <= +inf
   0 <= con_DCC_bin_y(1) <= 1
   0 <= con_DCC_bin_y(2) <= 1
   0 <= con_DCC_bin_y(3) <= 1
binary
  con_DCC_bin_y(1)
  con_DCC_bin_y(2)
  con_DCC_bin_y(3)
end 
