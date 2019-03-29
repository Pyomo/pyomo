ALL: petsc

CFLAGS = -I${ASL_BUILD}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

bindir = $(prefix)/bin

petsc: petsc.o fg_nl.o fg_dae.o scale.o printing.o chkopts
	${CLINKER} -o petsc petsc.o fg_nl.o fg_dae.o scale.o printing.o $(ASL_BUILD)/amplsolver.a ${PETSC_LIB}

install: petsc
	${CP} petsc ${bindir}
