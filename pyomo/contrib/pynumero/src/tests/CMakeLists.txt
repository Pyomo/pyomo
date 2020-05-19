
ADD_EXECUTABLE(pynumero_asl_test simple_test.cpp)
TARGET_LINK_LIBRARIES( pynumero_asl_test pynumero_ASL)
INSTALL(TARGETS pynumero_asl_test DESTINATION bin/tests )
INSTALL(FILES simple_nlp.nl DESTINATION bin/tests )
