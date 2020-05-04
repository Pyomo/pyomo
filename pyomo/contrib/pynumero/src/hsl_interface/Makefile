rootdir:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include $(rootdir)/Makefile.in

all: libpynumero_MA27.so libpynumero_MA57.so


ifeq ($(strip $(COINHSLDIR)),)

# If libcoinhsl is not available, link with the ma27d.o object one gets from
# compiling ma27, e.g. gfortran -O2 -fPIC -c -o ma27d.o ma27d.f
libpynumero_MA27.so: ma27Interface.o $(MA27DIR)/ma27d.o
	$(CL) $(CLFLAGS) $(OPTCL) $(OUTC) $@ $^ $(LIBGFORTRAN) $(LIBBLAS)

LIBMA57 = -L$(MA57DIR) -lma57

else

# IF libcoinhsl is available, get ma27 by dynamically linking libcoinhsl
libpynumero_MA27.so: ma27Interface.o
	$(CL) $(CLFLAGS) $(OPTCL) $(OUTC) $@ $^ $(LIBCOINHSL) $(LIBGFORTRAN) $(LIBBLAS)

LIBMA57 = $(LIBCOINHSL)

endif

ifeq ($(strip $(METISDIR)),)

LIBMETIS =

else

# This assumes metis is locally installed
# TODO: allow option to link metis installed to some system location
#       (and option to link system-installed ma57, for that matter)
LIBMETIS = -L$(METISDIR) -lmetis -lm

endif

libpynumero_MA57.so: ma57Interface.o
	$(CL) $(CLFLAGS) $(OPTCL) $(OUTC) $@ $^ $(LIBMA57) $(LIBGFORTRAN) $(LIBBLAS) $(LIBMETIS)

ma27Interface.o: ma27Interface.c
	$(CC) $(CCFLAGS) $(OPTCC) $(OUTC) $@ $^

ma57Interface.o: ma57Interface.c
	$(CC) $(CCFLAGS) $(OPTCC) $(OUTC) $@ $^

install: libpynumero_MA27.so libpynumero_MA57.so
	$(CP) $^ $(INSTALLDIR)/lib/

clean:
	$(RM) *27*.o *27*.so *57*.o *57*.so

