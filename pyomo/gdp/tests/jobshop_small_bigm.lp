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

c_e__gdp_bigm_relaxation_disj_xor(A_B_3)_:
+1 NoClash(A_B_3_0)_indicator_var
+1 NoClash(A_B_3_1)_indicator_var
= 1

c_e__gdp_bigm_relaxation_disj_xor(A_C_1)_:
+1 NoClash(A_C_1_0)_indicator_var
+1 NoClash(A_C_1_1)_indicator_var
= 1

c_e__gdp_bigm_relaxation_disj_xor(B_C_2)_:
+1 NoClash(B_C_2_0)_indicator_var
+1 NoClash(B_C_2_1)_indicator_var
= 1

c_u__pyomo_gdp_bigm_relaxation_relaxedDisjuncts(0)_NoClash(A_B_3_0)_c(ub)_:
+19 NoClash(A_B_3_0)_indicator_var
-1 t(A)
+1 t(B)
<= 19

c_u__pyomo_gdp_bigm_relaxation_relaxedDisjuncts(1)_NoClash(A_B_3_1)_c(ub)_:
+24 NoClash(A_B_3_1)_indicator_var
+1 t(A)
-1 t(B)
<= 19

c_u__pyomo_gdp_bigm_relaxation_relaxedDisjuncts(2)_NoClash(A_C_1_0)_c(ub)_:
+21 NoClash(A_C_1_0)_indicator_var
-1 t(A)
+1 t(C)
<= 19

c_u__pyomo_gdp_bigm_relaxation_relaxedDisjuncts(3)_NoClash(A_C_1_1)_c(ub)_:
+24 NoClash(A_C_1_1)_indicator_var
+1 t(A)
-1 t(C)
<= 19

c_u__pyomo_gdp_bigm_relaxation_relaxedDisjuncts(4)_NoClash(B_C_2_0)_c(ub)_:
+25 NoClash(B_C_2_0)_indicator_var
-1 t(B)
+1 t(C)
<= 19

c_u__pyomo_gdp_bigm_relaxation_relaxedDisjuncts(5)_NoClash(B_C_2_1)_c(ub)_:
+20 NoClash(B_C_2_1)_indicator_var
+1 t(B)
-1 t(C)
<= 19

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
    -inf <= ms <= +inf
   0 <= t(A) <= 19
   0 <= t(B) <= 19
   0 <= t(C) <= 19
   0 <= NoClash(A_B_3_0)_indicator_var <= 1
   0 <= NoClash(A_B_3_1)_indicator_var <= 1
   0 <= NoClash(A_C_1_0)_indicator_var <= 1
   0 <= NoClash(A_C_1_1)_indicator_var <= 1
   0 <= NoClash(B_C_2_0)_indicator_var <= 1
   0 <= NoClash(B_C_2_1)_indicator_var <= 1
binary
  NoClash(A_B_3_0)_indicator_var
  NoClash(A_B_3_1)_indicator_var
  NoClash(A_C_1_0)_indicator_var
  NoClash(A_C_1_1)_indicator_var
  NoClash(B_C_2_0)_indicator_var
  NoClash(B_C_2_1)_indicator_var
end
