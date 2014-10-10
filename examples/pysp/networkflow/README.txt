###########################################################################################
10-scenario instances 
###########################################################################################

CPLEX 10.1 EF values after 2 days of CPU time (values reported in CMS paper):

1ef10   152701.58 (1.32% optimality gap) - proved optimal by CPLEX 12.1 (16 threads, 3200 seconds real wall-clock time).
2ef10   148531.92 (1.54% optimality gap)
3ef10   148781.07 (1.13% optimality gap)
4ef10   159805.00 (2.15% optimality gap)
5ef10   153352.16 (1.68% optimality gap)

Using CPLEX 12.2 and PH, on Sandia's sleipnir server. Times are wall-clock time.

No fixing of continuous variables (or waiting for their convergence), as the resulting EF MIP is solvable.
Slamming after iteration 100, but not really - we haven't defined variable slam priorities. Cycle breaking 
when length exceeds 30.

Quadratic run results:

runph --model-directory=models --instance-directory=XXXX --max-iterations=100 --rho-cfgfile=config/rhosetter1.0.cfg --enable-ww-extensions --ww-extension-cfgfile=config/XXX.cfg --solve-ef 

Immediate fixing results (with wwph-immediatefixing.cfg).

1ef10: 7m5s,  solution=156611.92, 14 PH iterations (VERIFIED)
2ef10: 3m32s, solution=149138.58, 11 PH iterations (VERIFIED)
3ef10: 4m8s,  solution=151230.45,  9 PH iterations (VERIFIED)
4ef10: 3m32s, solution=161215.44, 12 PH iterations (VERIFIED)
5ef10: 3m6s,  solution=154286.86, 10 PH iterations (VERIFIED)

Fix lag = 10 (with wwph-fixlag10.cfg).

1ef10: 23m32s, solution=153542.68, 69 PH iterations (VERIFIED)
2ef10: Converges in 44 iterations. After solving remaining EF, objective=149048.82***OLD***
3ef10: 30m55s, solution=149973.90, 33 PH iterations
4ef10: Converges in 85 iterations. After solving remaining EF, objective=161147.9 - exhibits a lot of cycle-breaking.***OLD***
5ef10: Converges in 51 iterations. After solving remaining EF, objective=153279.42***OLD***

Fix lag = 20 (with wwph-fixlag20.cfg).

1ef10: Converges in 64 iterations.  After solving remaining EF, objective=153542.68***OLD***
2ef10: Converges in 110 iterations. After solving remaining EF, objective=149048.82***OLD***
3ef10: Converges in 67 iterations.  After solving remaining EF, objective=149973.9***OLD***
4ef10: ***FAILS TO CONVERGE - SLAMMING PRIORITIES ARE REQUIRED.***OLD***
5ef10: Converges in 81 iterations.  After solving remaining EF, objective=153279.42***OLD***

Linearized run results:

Same as above, but with the additional options: --bounds-cfgfile=config/xboundsetter.cfg --linearize-nonbinary-penalty-terms=8

Immediate fixing results (with wwph-immediatefixing.cfg).

1ef10: 6m58s, solution=156129.92,  15 PH iterations (VALIDATED)
2ef10: 3m23s, solution=150550.36, 10 PH iterations (VALIDATED)
3ef10: 5m40s, solution=151230.45, 19 PH iterations (VALIDATED)
4ef10: 2m38s, solution=161215.44,  7 PH iterations (VALIDATED)
5ef10: 3m26s, solution=154286.86, 12 PH iterations (VALIDATED)

Fix lag = 10 (with wwph-fixlag10.cfg).

1ef10:
2ef10:
3ef10:
4ef10: ***RUNNING***
5ef10:

Fix lag = 20 (with wwph-fixlag20.cfg).

1ef10:
2ef10:
3ef10:
4ef10:
5ef10:

###########################################################################################
50-scenario instances 
###########################################################################################

CPLEX 12.2 EF results:

1ef50: 
2ef50: 
3ef50:
4ef50: 
5ef50: 

Quadratic run results:

runph --model-directory=models --instance-directory=XXXX --max-iterations=100 --rho-cfgfile=config/rhosetter1.0.cfg --enable-ww-extensions --ww-extension-cfgfile=config/XXX.cfg --solve-ef 

Immediate fixing results (with wwph-immediatefixing.cfg).

1ef10: 372m51s, solution=161096.56, 54 PH iterations
2ef10: 
3ef10: 
4ef10: 
5ef10: 
