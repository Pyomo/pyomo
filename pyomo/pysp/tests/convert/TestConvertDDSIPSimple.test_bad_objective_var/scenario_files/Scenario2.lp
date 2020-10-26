\* Source Pyomo model name=Scenario2 *\

min 
x3:
+2 x1__DDSIP_FIRSTSTAGE
+1 x2
+0 ONE_VAR_CONSTANT

s.t.

c_l_x4_:
+1 x1__DDSIP_FIRSTSTAGE
>= 10

c_l_x5_:
+1 x2
>= 10

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
    -inf <= x1__DDSIP_FIRSTSTAGE <= +inf
    -inf <= x2 <= +inf
end
