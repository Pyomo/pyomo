\* Source Pyomo model name=unknown *\

min 
makespan:
+1 ms

s.t.

c_u_Feas(A)_:
-1 ms
+1 t(A)
<= -8

c_u_Feas(B)_:
-1 ms
+1 t(B)
<= -5

c_u_Feas(C)_:
-1 ms
+1 t(C)
<= -6

c_e__pyomo_gdp_bigm_reformulation_disj_xor(A_B_3)_:
+1 NoClash(A_B_3_0)_binary_indicator_var
+1 NoClash(A_B_3_1)_binary_indicator_var
= 1

c_e__pyomo_gdp_bigm_reformulation_disj_xor(A_C_1)_:
+1 NoClash(A_C_1_0)_binary_indicator_var
+1 NoClash(A_C_1_1)_binary_indicator_var
= 1

c_e__pyomo_gdp_bigm_reformulation_disj_xor(B_C_2)_:
+1 NoClash(B_C_2_0)_binary_indicator_var
+1 NoClash(B_C_2_1)_binary_indicator_var
= 1

c_u__pyomo_gdp_bigm_reformulation_relaxedDisjuncts(0)_transformedConstraints(c_0_None_ub)_:
-1 t(A)
+1 t(B)
+19.0 NoClash(A_B_3_0)_binary_indicator_var
<= 19.0

c_u__pyomo_gdp_bigm_reformulation_relaxedDisjuncts(1)_transformedConstraints(c_0_None_ub)_:
+1 t(A)
-1 t(B)
+24.0 NoClash(A_B_3_1)_binary_indicator_var
<= 19.0

c_u__pyomo_gdp_bigm_reformulation_relaxedDisjuncts(2)_transformedConstraints(c_0_None_ub)_:
-1 t(A)
+1 t(C)
+21.0 NoClash(A_C_1_0)_binary_indicator_var
<= 19.0

c_u__pyomo_gdp_bigm_reformulation_relaxedDisjuncts(3)_transformedConstraints(c_0_None_ub)_:
+1 t(A)
-1 t(C)
+24.0 NoClash(A_C_1_1)_binary_indicator_var
<= 19.0

c_u__pyomo_gdp_bigm_reformulation_relaxedDisjuncts(4)_transformedConstraints(c_0_None_ub)_:
-1 t(B)
+1 t(C)
+25.0 NoClash(B_C_2_0)_binary_indicator_var
<= 19.0

c_u__pyomo_gdp_bigm_reformulation_relaxedDisjuncts(5)_transformedConstraints(c_0_None_ub)_:
+1 t(B)
-1 t(C)
+20.0 NoClash(B_C_2_1)_binary_indicator_var
<= 19.0

bounds
   -inf <= ms <= +inf
   0 <= t(A) <= 19
   0 <= t(B) <= 19
   0 <= t(C) <= 19
   0 <= NoClash(A_B_3_0)_binary_indicator_var <= 1
   0 <= NoClash(A_B_3_1)_binary_indicator_var <= 1
   0 <= NoClash(A_C_1_0)_binary_indicator_var <= 1
   0 <= NoClash(A_C_1_1)_binary_indicator_var <= 1
   0 <= NoClash(B_C_2_0)_binary_indicator_var <= 1
   0 <= NoClash(B_C_2_1)_binary_indicator_var <= 1
binary
  NoClash(A_B_3_0)_binary_indicator_var
  NoClash(A_B_3_1)_binary_indicator_var
  NoClash(A_C_1_0)_binary_indicator_var
  NoClash(A_C_1_1)_binary_indicator_var
  NoClash(B_C_2_0)_binary_indicator_var
  NoClash(B_C_2_1)_binary_indicator_var
end
