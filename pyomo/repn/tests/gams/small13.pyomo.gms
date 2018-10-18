$offdigit

EQUATIONS
	c1
	c2
	c3
	c4;

VARIABLES
	GAMS_OBJECTIVE
	x1;

c1.. power(x1, 3) + (-1)*x1 =e= 0.0 ;
c2.. 10*(power(x1, 3) + (-1)*x1) =e= 0.0 ;
c3.. (power(x1, 3) + (-1)*x1)*0.1 =e= 0.0 ;
c4.. GAMS_OBJECTIVE =e= x1 ;

x1.l = 0.5;

MODEL GAMS_MODEL /all/ ;
SOLVE GAMS_MODEL USING nlp maximizing GAMS_OBJECTIVE;

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

