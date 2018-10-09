$offdigit

EQUATIONS
	c1_lo
	c2_hi
	c3_hi
	c4;

POSITIVE VARIABLES
	x1
	x2;

VARIABLES
	GAMS_OBJECTIVE
	x3;

c1_lo.. 2.0 =l= x1*x1 ;
c2_hi.. x1 + (-0.5)*x2 =l= 0.0 ;
c3_hi.. x3 - (x1 + 2.0) =l= 0.0 ;
c4.. GAMS_OBJECTIVE =e= x3 + x2*x2 + x1 ;

x3.lo = 7;

MODEL GAMS_MODEL /all/ ;
SOLVE GAMS_MODEL USING nlp minimizing GAMS_OBJECTIVE;

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

