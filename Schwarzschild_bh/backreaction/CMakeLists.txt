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

#Executable 5
set (PROG5 backreaction_N_Psi_bet3)
set (SOURCES5 src/${PROG5}.cpp)
add_executable(${PROG5} ${SOURCES5} ${HEADERS})
target_link_libraries(${PROG5} ${LIB_KADATH} ${PGPLOT_LIBRARIES} ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${LAPACK_LIBRARIES} ${SCALAPACK_LIBRARIES})

#Executable 6
set (PROG6 backreaction_bet3)
set (SOURCES6 src/${PROG6}.cpp)
add_executable(${PROG6} ${SOURCES6} ${HEADERS})
target_link_libraries(${PROG6} ${LIB_KADATH} ${PGPLOT_LIBRARIES} ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${LAPACK_LIBRARIES} ${SCALAPACK_LIBRARIES})

#Executable 7
set (PROG7 backreaction_N_Psi_ph12)
set (SOURCES7 src/${PROG7}.cpp)
add_executable(${PROG7} ${SOURCES7} ${HEADERS})
target_link_libraries(${PROG7} ${LIB_KADATH} ${PGPLOT_LIBRARIES} ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${LAPACK_LIBRARIES} ${SCALAPACK_LIBRARIES})

#Executable 8
set (PROG8 backreaction4_ph12)
set (SOURCES8 src/${PROG8}.cpp)
add_executable(${PROG8} ${SOURCES8} ${HEADERS})
target_link_libraries(${PROG8} ${LIB_KADATH} ${PGPLOT_LIBRARIES} ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${LAPACK_LIBRARIES} ${SCALAPACK_LIBRARIES})
