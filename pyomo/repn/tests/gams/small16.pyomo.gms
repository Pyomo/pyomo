$offdigit

EQUATIONS
	c1;

VARIABLES
	GAMS_OBJECTIVE
	x1
	x2;

c1.. GAMS_OBJECTIVE =e= x1 + x2 ;

x1.lo = 2;
x1.l = 1.0;
x2.lo = 1;
x2.l = 1.0;

MODEL GAMS_MODEL /all/ ;
SOLVE GAMS_MODEL USING lp minimizing GAMS_OBJECTIVE;

Scalars MODELSTAT 'model status', SOLVESTAT 'solve status';
MODELSTAT = GAMS_MODEL.modelstat;
SOLVESTAT = GAMS_MODEL.solvestat;

Scalar OBJEST 'best objective', OBJVAL 'objective value';
OBJEST = GAMS_MODEL.objest;
OBJVAL = GAMS_MODEL.objval;

Scalar NUMVAR 'number of variables';
NUMVAR = GAMS_MODEL.numvar

Scalar NUMEQU 'number of equations';
NUMEQU = GAMS_MODEL.numequ

Scalar NUMDVAR 'number of discrete variables';
NUMDVAR = GAMS_MODEL.numdvar

Scalar NUMNZ 'number of nonzeros';
NUMNZ = GAMS_MODEL.numnz

Scalar ETSOLVE 'time to execute solve statement';
ETSOLVE = GAMS_MODEL.etsolve

