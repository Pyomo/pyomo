\* Problem: test3 *\

Minimize
 cost: + x(Wheat) + x(Cornmeal) + x(Cannedmilk) + x(Margarine)
 + x(Cheese) + x(Peanut~B) + x(Lard) + x(Liver) + x(Porkroast)
 + x(Salmon) + x(Greenbeans) + x(Cabbage) + x(Onions) + x(Potatoes)
 + x(Spinach) + x(Sweet~Pot) + x(Peaches) + x(Prunes) + x(Limabeans)
 + x(Navybeans)

Subject To
 nb(Calorie): + 44.7 x(Wheat) + 36 x(Cornmeal) + 8.4 x(Cannedmilk)
 + 20.6 x(Margarine) + 7.4 x(Cheese) + 15.7 x(Peanut~B) + 41.7 x(Lard)
 + 2.2 x(Liver) + 4.4 x(Porkroast) + 5.8 x(Salmon) + 2.4 x(Greenbeans)
 + 2.6 x(Cabbage) + 5.8 x(Onions) + 14.3 x(Potatoes) + 1.1 x(Spinach)
 + 9.6 x(Sweet~Pot) + 8.5 x(Peaches) + 12.8 x(Prunes)
 + 17.4 x(Limabeans) + 26.9 x(Navybeans) = 3
 nb(Protein): + 1411 x(Wheat) + 897 x(Cornmeal) + 422 x(Cannedmilk)
 + 17 x(Margarine) + 448 x(Cheese) + 661 x(Peanut~B) + 333 x(Liver)
 + 249 x(Porkroast) + 705 x(Salmon) + 138 x(Greenbeans) + 125 x(Cabbage)
 + 166 x(Onions) + 336 x(Potatoes) + 106 x(Spinach) + 138 x(Sweet~Pot)
 + 87 x(Peaches) + 99 x(Prunes) + 1055 x(Limabeans) + 1691 x(Navybeans)
 = 70
 nb(Calcium): + 2 x(Wheat) + 1.7 x(Cornmeal) + 15.1 x(Cannedmilk)
 + 0.6 x(Margarine) + 16.4 x(Cheese) + x(Peanut~B) + 0.2 x(Liver)
 + 0.3 x(Porkroast) + 6.8 x(Salmon) + 3.7 x(Greenbeans) + 4 x(Cabbage)
 + 3.8 x(Onions) + 1.8 x(Potatoes) + 2.7 x(Sweet~Pot) + 1.7 x(Peaches)
 + 2.5 x(Prunes) + 3.7 x(Limabeans) + 11.4 x(Navybeans) = 0.8
 nb(Iron): + 365 x(Wheat) + 99 x(Cornmeal) + 9 x(Cannedmilk)
 + 6 x(Margarine) + 19 x(Cheese) + 48 x(Peanut~B) + 139 x(Liver)
 + 37 x(Porkroast) + 45 x(Salmon) + 80 x(Greenbeans) + 36 x(Cabbage)
 + 59 x(Onions) + 118 x(Potatoes) + 138 x(Spinach) + 54 x(Sweet~Pot)
 + 173 x(Peaches) + 154 x(Prunes) + 459 x(Limabeans) + 792 x(Navybeans)
 = 12
 nb(Vitamin~A): + 30.9 x(Cornmeal) + 26 x(Cannedmilk)
 + 55.8 x(Margarine) + 28.1 x(Cheese) + 0.2 x(Lard) + 169.2 x(Liver)
 + 3.5 x(Salmon) + 69 x(Greenbeans) + 7.2 x(Cabbage) + 16.6 x(Onions)
 + 6.7 x(Potatoes) + 918.4 x(Spinach) + 290.7 x(Sweet~Pot)
 + 86.8 x(Peaches) + 85.7 x(Prunes) + 5.1 x(Limabeans) = 5
 nb(Vitamin~B1): + 55.4 x(Wheat) + 17.4 x(Cornmeal) + 3 x(Cannedmilk)
 + 0.2 x(Margarine) + 0.8 x(Cheese) + 9.6 x(Peanut~B) + 6.4 x(Liver)
 + 18.2 x(Porkroast) + x(Salmon) + 4.3 x(Greenbeans) + 9 x(Cabbage)
 + 4.7 x(Onions) + 29.4 x(Potatoes) + 5.7 x(Spinach) + 8.4 x(Sweet~Pot)
 + 1.2 x(Peaches) + 3.9 x(Prunes) + 26.9 x(Limabeans)
 + 38.4 x(Navybeans) = 1.8
 nb(Vitamin~B2): + 33.3 x(Wheat) + 7.9 x(Cornmeal) + 23.5 x(Cannedmilk)
 + 10.3 x(Cheese) + 8.1 x(Peanut~B) + 0.5 x(Lard) + 50.8 x(Liver)
 + 3.6 x(Porkroast) + 4.9 x(Salmon) + 5.8 x(Greenbeans) + 4.5 x(Cabbage)
 + 5.9 x(Onions) + 7.1 x(Potatoes) + 13.8 x(Spinach) + 5.4 x(Sweet~Pot)
 + 4.3 x(Peaches) + 4.3 x(Prunes) + 38.2 x(Limabeans)
 + 24.6 x(Navybeans) = 2.7
 nb(Niacin): + 441 x(Wheat) + 106 x(Cornmeal) + 11 x(Cannedmilk)
 + 4 x(Cheese) + 471 x(Peanut~B) + 5 x(Lard) + 316 x(Liver)
 + 79 x(Porkroast) + 209 x(Salmon) + 37 x(Greenbeans) + 26 x(Cabbage)
 + 21 x(Onions) + 198 x(Potatoes) + 33 x(Spinach) + 83 x(Sweet~Pot)
 + 55 x(Peaches) + 65 x(Prunes) + 93 x(Limabeans) + 217 x(Navybeans)
 = 18
 nb(Vitamin~C): + 60 x(Cannedmilk) + 525 x(Liver) + 862 x(Greenbeans)
 + 5369 x(Cabbage) + 1184 x(Onions) + 2522 x(Potatoes) + 2755 x(Spinach)
 + 1912 x(Sweet~Pot) + 57 x(Peaches) + 257 x(Prunes) = 75

End
