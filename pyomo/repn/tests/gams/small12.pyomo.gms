$offdigit

EQUATIONS
	c1
	c2
	c3
	c4
	c5
	c6
	c7
	c8
	c9
	c10
	c11
	c12
	c13
	c14
	c15
	c16
	c17
	c18
	c19
	c20
	c21
	c22
	c23
	c24
	c25;

VARIABLES
	GAMS_OBJECTIVE
	x1
	x2
	x3
	x4
	x5
	x6
	x7;

c1.. Expr_if( ( 0.0 ), then=( vTrue ), else=( vFalse ) ) =e= -1 ;
c2.. Expr_if( ( 1.0 ), then=( vTrue ), else=( vFalse ) ) =e= 1 ;
c3.. Expr_if( ( vN1  <=  0.0 ), then=( vTrue ), else=( vFalse ) ) =e= 1 ;
c4.. Expr_if( ( v0  <=  0.0 ), then=( vTrue ), else=( vFalse ) ) =e= 1 ;
c5.. Expr_if( ( vP1  <=  0.0 ), then=( vTrue ), else=( vFalse ) ) =e= -1 ;
c6.. Expr_if( ( vN1  <  0.0 ), then=( vTrue ), else=( vFalse ) ) =e= 1 ;
c7.. Expr_if( ( v0  <  0.0 ), then=( vTrue ), else=( vFalse ) ) =e= -1 ;
c8.. Expr_if( ( vP1  <  0.0 ), then=( vTrue ), else=( vFalse ) ) =e= -1 ;
c9.. Expr_if( ( 0.0  <=  10.0*vN1 ), then=( vTrue ), else=( vFalse ) ) =e= -1 ;
c10.. Expr_if( ( 0.0  <=  10.0*v0 ), then=( vTrue ), else=( vFalse ) ) =e= 1 ;
c11.. Expr_if( ( 0.0  <=  10.0*vP1 ), then=( vTrue ), else=( vFalse ) ) =e= 1 ;
c12.. Expr_if( ( 0.0  <  10.0*vN1 ), then=( vTrue ), else=( vFalse ) ) =e= -1 ;
c13.. Expr_if( ( 0.0  <  10.0*v0 ), then=( vTrue ), else=( vFalse ) ) =e= -1 ;
c14.. Expr_if( ( 0.0  <  10.0*vP1 ), then=( vTrue ), else=( vFalse ) ) =e= 1 ;
c15.. Expr_if( ( -1  <=  vN2  <=  1 ), then=( vTrue ), else=( vFalse ) ) =e= -1 ;
c16.. Expr_if( ( - vP1  <=  vN1  <=  1 ), then=( vTrue ), else=( vFalse ) ) =e= 1 ;
c17.. Expr_if( ( - vP1**2  <=  v0  <=  1 ), then=( vTrue ), else=( vFalse ) ) =e= 1 ;
c18.. Expr_if( ( vN1  <=  vP1  <=  1 ), then=( vTrue ), else=( vFalse ) ) =e= 1 ;
c19.. Expr_if( ( -1  <=  vP2  <=  1 ), then=( vTrue ), else=( vFalse ) ) =e= -1 ;
c20.. Expr_if( ( -1  <  vN2  <  1 ), then=( vTrue ), else=( vFalse ) ) =e= -1 ;
c21.. Expr_if( ( -1  <  vN1  <  vP1 ), then=( vTrue ), else=( vFalse ) ) =e= -1 ;
c22.. Expr_if( ( -1  <  v0  <  vP1**2 ), then=( vTrue ), else=( vFalse ) ) =e= 1 ;
c23.. Expr_if( ( -1  <  vP1  <  vP1 ), then=( vTrue ), else=( vFalse ) ) =e= -1 ;
c24.. Expr_if( ( -1  <  vP2  <  1 ), then=( vTrue ), else=( vFalse ) ) =e= -1 ;
c25.. GAMS_OBJECTIVE =e= 10*Expr_if( ( v0 ), then=( vTrue ), else=( vFalse ) ) ;

x1.l = 1;
x2.l = -1;
x3.l = -1;
x4.l = 0;
x5.l = 1;
x6.l = -2;
x7.l = 2;

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

