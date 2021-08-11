cmake_minimum_required(VERSION 2.8)
project(br)

include ($ENV{HOME_KADATH}/Cmake/CMakeExec.cmake)

#Executable 1
set (PROG1 backreaction_full)
set (SOURCES1 src/${PROG1}.cpp)
add_executable(${PROG1} ${SOURCES1} ${HEADERS})
target_link_libraries(${PROG1} ${LIB_KADATH} ${PGPLOT_LIBRARIES} ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${LAPACK_LIBRARIES} ${SCALAPACK_LIBRARIES})

#Executable 2
set (PROG2 read_backreaction)
set (SOURCES2 src/${PROG2}.cpp)
add_executable(${PROG2} ${SOURCES2} ${HEADERS})
target_link_libraries(${PROG2} ${LIB_KADATH} ${PGPLOT_LIBRARIES} ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${LAPACK_LIBRARIES} ${SCALAPACK_LIBRARIES})


#Executable 3
set (PROG3 backreaction_N_Psi)
set (SOURCES3 src/${PROG3}.cpp)
add_executable(${PROG3} ${SOURCES3} ${HEADERS})
target_link_libraries(${PROG3} ${LIB_KADATH} ${PGPLOT_LIBRARIES} ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${LAPACK_LIBRARIES} ${SCALAPACK_LIBRARIES})

#Executable 4
set (PROG4 backreaction_N_Psi_bet)
set (SOURCES4 src/${PROG4}.cpp)
add_executable(${PROG4} ${SOURCES4} ${HEADERS})
target_link_libraries(${PROG4} ${LIB_KADATH} ${PGPLOT_LIBRARIES} ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${LAPACK_LIBRARIES} ${SCALAPACK_LIBRARIES})