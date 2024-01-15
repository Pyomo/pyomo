\* Source Pyomo model name=unknown *\

min 
x1:
+1 x2
+2 x3

s.t.

c_l_x4_:
+3 x2
+4 x3
>= 1

c_l_x5_:
+2 x2
+5 x3
>= 2

bounds
   0 <= x2 <= +inf
   0 <= x3 <= +inf
general
  x2
  x3
end
