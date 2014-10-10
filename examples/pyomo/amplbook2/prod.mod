# Sets
set P;

# Parameters
param a {j in P};
param b;
param c {j in P};
param u {j in P};

# Variables
var X {j in P};

# Objective
maximize Total_Profit: sum {j in P} c[j] * X[j];

# Time Constraint
subject to Time: sum {j in P} (1/a[j]) * X[j] <= b;

# Limit Constraint
subject to Limit {j in P}: 0 <= X[j] <= u[j];
