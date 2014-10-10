# Create a large bin-packing problem to make cplex run a while
# so we can test how it handles interrupts.

param N default 99;
set I = 1..N;
var b{I} binary;

maximize zot: sum{i in I} (1 + .02*i)*b[i];

param w{i in 1 .. N - 9, j in i .. i + 9};# = Uniform(1,2);

param rhs{1 .. N-9};

s.t. bletch{i in 1 .. N - 9}:
	sum{j in i .. i+9} w[i,j]*b[j] <= rhs[i]; #Uniform(5,10);
