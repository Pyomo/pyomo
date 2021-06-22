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

c_e__pyomo_gdp_hull_reformulation_disaggregationConstraints(0_A_B_3)_:
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(0)_disaggregatedVars__t(B)_
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(1)_disaggregatedVars__t(B)_
+1 t(B)
= 0

c_e__pyomo_gdp_hull_reformulation_disaggregationConstraints(0_A_C_1)_:
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(2)_disaggregatedVars__t(C)_
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(3)_disaggregatedVars__t(C)_
+1 t(C)
= 0

c_e__pyomo_gdp_hull_reformulation_disaggregationConstraints(0_B_C_2)_:
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(4)_disaggregatedVars__t(C)_
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(5)_disaggregatedVars__t(C)_
+1 t(C)
= 0

c_e__pyomo_gdp_hull_reformulation_disaggregationConstraints(1_A_B_3)_:
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(0)_disaggregatedVars__t(A)_
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(1)_disaggregatedVars__t(A)_
+1 t(A)
= 0

c_e__pyomo_gdp_hull_reformulation_disaggregationConstraints(1_A_C_1)_:
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(2)_disaggregatedVars__t(A)_
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(3)_disaggregatedVars__t(A)_
+1 t(A)
= 0

c_e__pyomo_gdp_hull_reformulation_disaggregationConstraints(1_B_C_2)_:
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(4)_disaggregatedVars__t(B)_
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(5)_disaggregatedVars__t(B)_
+1 t(B)
= 0

c_e__pyomo_gdp_hull_reformulation_disj_xor(A_B_3)_:
+1 NoClash(A_B_3_0)_binary_indicator_var
+1 NoClash(A_B_3_1)_binary_indicator_var
= 1

c_e__pyomo_gdp_hull_reformulation_disj_xor(A_C_1)_:
+1 NoClash(A_C_1_0)_binary_indicator_var
+1 NoClash(A_C_1_1)_binary_indicator_var
= 1

c_e__pyomo_gdp_hull_reformulation_disj_xor(B_C_2)_:
+1 NoClash(B_C_2_0)_binary_indicator_var
+1 NoClash(B_C_2_1)_binary_indicator_var
= 1

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(0)__t(B)_bounds_(ub)_:
-19 NoClash(A_B_3_0)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(0)_disaggregatedVars__t(B)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(0)__t(A)_bounds_(ub)_:
-19 NoClash(A_B_3_0)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(0)_disaggregatedVars__t(A)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(0)__NoClash(A_B_3_0)_c_(ub)_:
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(0)_disaggregatedVars__t(A)_
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(0)_disaggregatedVars__t(B)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(1)__t(B)_bounds_(ub)_:
-19 NoClash(A_B_3_1)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(1)_disaggregatedVars__t(B)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(1)__t(A)_bounds_(ub)_:
-19 NoClash(A_B_3_1)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(1)_disaggregatedVars__t(A)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(1)__NoClash(A_B_3_1)_c_(ub)_:
+5 NoClash(A_B_3_1)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(1)_disaggregatedVars__t(A)_
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(1)_disaggregatedVars__t(B)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(2)__t(C)_bounds_(ub)_:
-19 NoClash(A_C_1_0)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(2)_disaggregatedVars__t(C)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(2)__t(A)_bounds_(ub)_:
-19 NoClash(A_C_1_0)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(2)_disaggregatedVars__t(A)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(2)__NoClash(A_C_1_0)_c_(ub)_:
+2 NoClash(A_C_1_0)_binary_indicator_var
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(2)_disaggregatedVars__t(A)_
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(2)_disaggregatedVars__t(C)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(3)__t(C)_bounds_(ub)_:
-19 NoClash(A_C_1_1)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(3)_disaggregatedVars__t(C)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(3)__t(A)_bounds_(ub)_:
-19 NoClash(A_C_1_1)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(3)_disaggregatedVars__t(A)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(3)__NoClash(A_C_1_1)_c_(ub)_:
+5 NoClash(A_C_1_1)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(3)_disaggregatedVars__t(A)_
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(3)_disaggregatedVars__t(C)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(4)__t(C)_bounds_(ub)_:
-19 NoClash(B_C_2_0)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(4)_disaggregatedVars__t(C)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(4)__t(B)_bounds_(ub)_:
-19 NoClash(B_C_2_0)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(4)_disaggregatedVars__t(B)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(4)__NoClash(B_C_2_0)_c_(ub)_:
+6 NoClash(B_C_2_0)_binary_indicator_var
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(4)_disaggregatedVars__t(B)_
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(4)_disaggregatedVars__t(C)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(5)__t(C)_bounds_(ub)_:
-19 NoClash(B_C_2_1)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(5)_disaggregatedVars__t(C)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(5)__t(B)_bounds_(ub)_:
-19 NoClash(B_C_2_1)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(5)_disaggregatedVars__t(B)_
<= 0

c_u__pyomo_gdp_hull_reformulation_relaxedDisjuncts(5)__NoClash(B_C_2_1)_c_(ub)_:
+1 NoClash(B_C_2_1)_binary_indicator_var
+1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(5)_disaggregatedVars__t(B)_
-1 _pyomo_gdp_hull_reformulation_relaxedDisjuncts(5)_disaggregatedVars__t(C)_
<= 0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
    -inf <= ms <= +inf
   0 <= t(A) <= 19
   0 <= t(B) <= 19
   0 <= t(C) <= 19
   0 <= NoClash(A_B_3_0)_binary_indicator_var <= 1
   0 <= _pyomo_gdp_hull_reformulation_relaxedDisjuncts(0)_disaggregatedVars__t(B)_ <= 19
   0 <= _pyomo_gdp_hull_reformulation_relaxedDisjuncts(0)_disaggregatedVars__t(A)_ <= 19
   0 <= NoClash(A_B_3_1)_binary_indicator_var <= 1
   0 <= _pyomo_gdp_hull_reformulation_relaxedDisjuncts(1)_disaggregatedVars__t(B)_ <= 19
   0 <= _pyomo_gdp_hull_reformulation_relaxedDisjuncts(1)_disaggregatedVars__t(A)_ <= 19
   0 <= NoClash(A_C_1_0)_binary_indicator_var <= 1
   0 <= _pyomo_gdp_hull_reformulation_relaxedDisjuncts(2)_disaggregatedVars__t(C)_ <= 19
   0 <= _pyomo_gdp_hull_reformulation_relaxedDisjuncts(2)_disaggregatedVars__t(A)_ <= 19
   0 <= NoClash(A_C_1_1)_binary_indicator_var <= 1
   0 <= _pyomo_gdp_hull_reformulation_relaxedDisjuncts(3)_disaggregatedVars__t(C)_ <= 19
   0 <= _pyomo_gdp_hull_reformulation_relaxedDisjuncts(3)_disaggregatedVars__t(A)_ <= 19
   0 <= NoClash(B_C_2_0)_binary_indicator_var <= 1
   0 <= _pyomo_gdp_hull_reformulation_relaxedDisjuncts(4)_disaggregatedVars__t(C)_ <= 19
   0 <= _pyomo_gdp_hull_reformulation_relaxedDisjuncts(4)_disaggregatedVars__t(B)_ <= 19
   0 <= NoClash(B_C_2_1)_binary_indicator_var <= 1
   0 <= _pyomo_gdp_hull_reformulation_relaxedDisjuncts(5)_disaggregatedVars__t(C)_ <= 19
   0 <= _pyomo_gdp_hull_reformulation_relaxedDisjuncts(5)_disaggregatedVars__t(B)_ <= 19
binary
  NoClash(A_B_3_0)_binary_indicator_var
  NoClash(A_B_3_1)_binary_indicator_var
  NoClash(A_C_1_0)_binary_indicator_var
  NoClash(A_C_1_1)_binary_indicator_var
  NoClash(B_C_2_0)_binary_indicator_var
  NoClash(B_C_2_1)_binary_indicator_var
end
