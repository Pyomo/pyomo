# STIGLER'S NUTRITION MODEL
#
# This model determines a least cost diet which meets the daily
# allowances of nutrients for a moderately active man weighing 154 lbs.
#
#  References:
#              Dantzig G B, "Linear Programming and Extensions."
#              Princeton University Press, Princeton, New Jersey, 1963,
#              Chapter 27-1.

set N;
/* nutrients */

set F;
/* foods */

param b{N};
/* required daily allowances of nutrients */

param a{F,N};
/* nutritive value of foods (per dollar spent) */

var x{f in F} >= 0;
/* dollars of food f to be purchased daily */

s.t. nb{n in N}: sum{f in F} a[f,n] * x[f] = b[n];
/* nutrient balance (units) */

minimize cost: sum{f in F} x[f];
/* total food bill (dollars) */

data;

param : N : b :=
         Calorie       3 /* thousands */
         Protein      70 /* grams */
         Calcium     0.8 /* grams */
         Iron         12 /* milligrams */
         Vitamin-A     5 /* thousands IUs */
         Vitamin-B1  1.8 /* milligrams */
         Vitamin-B2  2.7 /* milligrams */
         Niacin       18 /* milligrams */
         Vitamin-C    75 /* milligrams */  ;

set F := Wheat Cornmeal Cannedmilk Margarine Cheese Peanut-B Lard
         Liver Porkroast Salmon Greenbeans Cabbage Onions Potatoes
         Spinach Sweet-Pot Peaches Prunes Limabeans Navybeans;

param a default 0

:           Calorie  Protein  Calcium  Iron  Vitamin-A  Vitamin-B1 :=
#            (1000)    (g)      (g)    (mg)   (1000IU)     (mg)

Wheat         44.7     1411      2.0    365        .       55.4
Cornmeal      36        897      1.7     99      30.9      17.4
Cannedmilk     8.4      422     15.1      9      26         3
Margarine     20.6       17       .6      6      55.8        .2
Cheese         7.4      448     16.4     19      28.1        .8
Peanut-B      15.7      661      1       48        .        9.6
Lard          41.7        .       .       .        .2        .
Liver          2.2      333       .2    139     169.2       6.4
Porkroast      4.4      249       .3     37        .       18.2
Salmon         5.8      705      6.8     45       3.5       1
Greenbeans     2.4      138      3.7     80      69         4.3
Cabbage        2.6      125      4       36       7.2       9
Onions         5.8      166      3.8     59      16.6       4.7
Potatoes      14.3      336      1.8    118       6.7      29.4
Spinach        1.1      106       .     138     918.4       5.7
Sweet-Pot      9.6      138      2.7     54     290.7       8.4
Peaches        8.5       87      1.7    173      86.8       1.2
Prunes        12.8       99      2.5    154      85.7       3.9
Limabeans     17.4     1055      3.7    459       5.1      26.9
Navybeans     26.9     1691     11.4    792        .       38.4

:          Vitamin-B2  Niacin  Vitamin-C :=
#             (mg)      (mg)     (mg)

Wheat         33.3       441         .
Cornmeal       7.9       106         .
Cannedmilk    23.5        11        60
Margarine       .          .         .
Cheese        10.3         4         .
Peanut-B       8.1       471         .
Lard            .5         5         .
Liver         50.8       316       525
Porkroast      3.6        79         .
Salmon         4.9       209         .
Greenbeans     5.8        37       862
Cabbage        4.5        26      5369
Onions         5.9        21      1184
Potatoes       7.1       198      2522
Spinach       13.8        33      2755
Sweet-Pot      5.4        83      1912
Peaches        4.3        55        57
Prunes         4.3        65       257
Limabeans     38.2        93         .
Navybeans     24.6       217         .   ;

end;
