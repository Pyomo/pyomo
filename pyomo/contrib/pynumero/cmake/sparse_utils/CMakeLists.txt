ADD_LIBRARY( pynumero_SPARSE SHARED src/SparseUtils.cpp)
INSTALL(TARGETS pynumero_SPARSE
        DESTINATION lib
       )
INSTALL(DIRECTORY src
        DESTINATION include
       )
