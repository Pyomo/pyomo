PyNumero libraries
==================

Pynumero relies on C/C++ extensions for expensive computing operations. This folder contain the C/C++ code to build the libraires.

Instructions:

# if conda is not available
cd third_party/ASL
./get.ASL
cd solvers
./configurehere
make # remove -DNo_dtoa from cflags in makefile
cd ../../
mkdir build
cd build
cmake ..
make
cp asl_interface/libpynumero_ASL* ../../extensions/lib/<OSNAME>
cp sparse_utils/libpynumero_SPARSE* ../../extensions/lib/<OSNAME>

# if conda is available and want to link to ASL in ampl-mp
conda install -c conda-forge ampl-mp
mkdir build
cd build
cmake .. -DMP_PATH=<PATH_TO_ampl-mp>
make
cp asl_interface/libpynumero_ASL* ../../extensions/lib/<OSNAME>
cp sparse_utils/libpynumero_SPARSE* ../../extensions/lib/<OSNAME>

# if conda available and do not want to compile
conda install -c conda-forge pynumero_libraries

# Note: by default libraries are linked dynamically to stdlib. To link statically enable option -DSTATIC_LINK=ON 




