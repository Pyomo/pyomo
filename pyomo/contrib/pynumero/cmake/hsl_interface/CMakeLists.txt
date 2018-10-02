
enable_language(Fortran)
set(DEFAULT_COIN_HSL_PATH "${PROJECT_SOURCE_DIR}/third_party/HSL/coinhsl" )

# get the coinhsl library
if (USE_COIN_HSL_PATH)
    ADD_LIBRARY( coinhsl STATIC
            "${USE_COIN_HSL_PATH}/ma27/ma27d.f"
            #"${USE_COIN_HSL_PATH}/ma57/ma57d.f"
            )
    set( COIN_HSL_INCLUDE_DIRS "${USE_COIN_HSL_PATH}/ma27" )
    #   set_property(TARGET coinhsl PROPERTY LINKER_LANGUAGE Fortran)
    set(COIN_HSL_LIBRARIES coinhsl)
    set( WITH_HSL true)
elseif(DEFAULT_COIN_HSL_PATH)
    ADD_LIBRARY( coinhsl STATIC
            "${DEFAULT_COIN_HSL_PATH}/ma27/ma27d.f"
            #"${DEFAULT_COIN_HSL_PATH}/ma57/ma57d.f"
            )
    SET_PROPERTY(TARGET coinhsl PROPERTY POSITION_INDEPENDENT_CODE ON)    
    set(COIN_HSL_INCLUDE_DIRS "${DEFAULT_COIN_HSL_PATH}/ma27" )
    set(COIN_HSL_LIBRARIES coinhsl)
    set( WITH_HSL true)
else()
    # for now throw an error - when we support more linear solvers, we can
    # allow this to proceed
    message( FATAL_ERROR "This software currently requires MA27 provided by coinhsl. Please set USE_COIN_HSL_PATH to the location of coinhsl codes." )
    set(COIN_HSL_LIRARIES "")

    MESSAGE( STATUS "*** HSL support not included - set USE_COIN_HSL_PATH to compile in HSL support." )
    set( COIN_HSL_INCLUDE_DIRS "" )
    set( WITH_HSL false)
endif()


if (${WITH_HSL})
    MESSAGE("hsl includes ${COIN_HSL_INCLUDE_DIRS}")
    ADD_LIBRARY(pynumero_HSL MODULE src/ma27Interface.cpp)
    SET_PROPERTY(TARGET pynumero_HSL PROPERTY POSITION_INDEPENDENT_CODE ON)
    TARGET_INCLUDE_DIRECTORIES( pynumero_HSL PUBLIC ${COIN_HSL_INCLUDE_DIRS} )
    TARGET_LINK_LIBRARIES( pynumero_HSL ${COIN_HSL_LIBRARIES} ${CMAKE_DL_LIBS} gfortran)
endif()
