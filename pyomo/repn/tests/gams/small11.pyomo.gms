$offdigit

EQUATIONS
	c1_lo
	c1_hi
	c2_lo
	c2_hi
	c3_lo
	c3_hi
	c4;

VARIABLES
	GAMS_OBJECTIVE
	x1
	x2
	x3;

c1_lo.. -1 =l= 1 ;
c1_hi.. 1 =l= 1 ;
c2_lo.. -1 =l= x1 ;
c2_hi.. x1 =l= 1 ;
c3_lo.. -1 =l= x2 ;
c3_hi.. x2 =l= 1 ;
c4.. GAMS_OBJECTIVE =e= x3 ;


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

