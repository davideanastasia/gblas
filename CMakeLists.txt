cmake_minimum_required(VERSION 2.6)

#SET(CMAKE_BUILD_TYPE "Release")
#use " cmake path_to_project_srcs -DCMAKE_BUILD_TYPE=RELEASE "

project(gblas)

add_definitions("-fomit-frame-pointer -msse3 -ffast-math")
add_definitions("-Wall")
include_directories("${CMAKE_SOURCE_DIR}/include")
# -Wall -mfpmath=sse -ffast-math

#OpenMP
#IF(OMP)
#FIND_PACKAGE(OpenMP REQUIRED)
#SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
#SET(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
#MESSAGE("OpenMP flags = "${OpenMP_CXX_FLAGS})
#ENDIF(OMP)

#condition choice of the right linked library
include_directories("${CMAKE_SOURCE_DIR}/goto")
#IF(APPLE)
#Apple
SET(GOTO_LIB "-L${CMAKE_SOURCE_DIR}/goto/osx/ -lgoto2")
#ELSE(APPLE)
#Linux
#SET(GOTO_LIB "-L${CMAKE_SOURCE_DIR}/goto/linux_64/ -lgoto2") # ?!?
#ENDIF(APPLE)
#FIND_PACKAGE(Threads REQUIRED)

SET(GBLAS_SRC_FILES
	include/gblas.h
	include/gblas_matrix_utils.h
	src/gblas_matrix_utils.cpp
	include/gblas_sse_utils.h
	src/gblas_sse_utils.cpp
	include/gblas_quantizer.h
	src/gblas_quantizer.cpp
	include/gblas_stat_model.h
	src/gblas_stat_model.cpp
	include/gblas_kernels.h
	src/gblas_kernels_d_packed.cpp
	src/gblas_kernels_d_std.cpp
	src/gblas_kernels_s_packed.cpp
	src/gblas_kernels_s_std.cpp
	src/gblas_dgemm_packed.cpp
	src/gblas_dgemm_std.cpp
	src/gblas_sgemm_packed.cpp
	src/gblas_sgemm_std.cpp
)

# GBLAS final library
add_library(gblas ${GBLAS_SRC_FILES})

# GBLAS fixed companders (model check)
add_library(gblas_fixed_companders ${GBLAS_SRC_FILES})
set_property(
  TARGET gblas_fixed_companders
  APPEND PROPERTY COMPILE_DEFINITIONS FIXED_COMPOUNDERS
)

# GBLAS fixed companders (model check)
add_library(gblas_external_aquant ${GBLAS_SRC_FILES})
set_property(
  TARGET gblas_external_aquant
  APPEND PROPERTY COMPILE_DEFINITIONS EXTERNAL_A_QUANT
)
  
SET(TEST_COMMON_SRC_FILES
 	tests/matrix_utils.h
 	tests/matrix_utils_t.h
	tests/matrix_utils.cpp
 	tests/high_priority_process.h
	tests/high_priority_process.cpp	
	tests/msec_timer.h
	tests/msec_timer.cpp
)  
  
add_executable(test_gblas
	tests/main.cpp
 	tests/perf_tests.h
	tests/perf_tests.cpp
  	${TEST_COMMON_SRC_FILES}
)
target_link_libraries(test_gblas gblas ${GOTO_LIB} ${CMAKE_THREAD_LIBS_INIT})

# 2dpca source code test
add_executable(gblas_2dpca
        tests/gblas_2dpca.cpp
        ${TEST_COMMON_SRC_FILES}
)
target_link_libraries(gblas_2dpca gblas)


# Throughput Tests
add_executable(throughput_dgemm
	tests/throughput_dgemm.cpp
  	${TEST_COMMON_SRC_FILES}
)
target_link_libraries(throughput_dgemm gblas ${GOTO_LIB})

add_executable(throughput_sgemm
	tests/throughput_sgemm.cpp
  	${TEST_COMMON_SRC_FILES})
target_link_libraries(throughput_sgemm gblas ${GOTO_LIB})

# Numerical Representation Tests
add_executable(num_repr_sgemm
	tests/num_repr_sgemm.cpp
  	${TEST_COMMON_SRC_FILES})
set_property(
  TARGET num_repr_sgemm
  APPEND PROPERTY COMPILE_DEFINITIONS FIXED_COMPOUNDERS
)
target_link_libraries(num_repr_sgemm gblas_fixed_companders ${GOTO_LIB})

add_executable(num_repr_sgemm_20110905
        tests/num_repr_sgemm_20110905.cpp
        ${TEST_COMMON_SRC_FILES})
set_property(
  TARGET num_repr_sgemm_20110905
  APPEND PROPERTY COMPILE_DEFINITIONS FIXED_COMPOUNDERS
)
target_link_libraries(num_repr_sgemm_20110905 gblas_fixed_companders ${GOTO_LIB})

add_executable(num_repr_dgemm
	tests/num_repr_dgemm.cpp
  	${TEST_COMMON_SRC_FILES})
set_property(
  TARGET num_repr_dgemm
  APPEND PROPERTY COMPILE_DEFINITIONS FIXED_COMPOUNDERS
)
target_link_libraries(num_repr_dgemm gblas_fixed_companders ${GOTO_LIB})


#add_executable(num_repr_dgemm_20110905
#        tests/num_repr_dgemm_20110905.cpp
#        ${TEST_COMMON_SRC_FILES})
#set_property(
#  TARGET num_repr_dgemm_20110905
#  APPEND PROPERTY COMPILE_DEFINITIONS FIXED_COMPOUNDERS
#)
#target_link_libraries(num_repr_dgemm_20110905 gblas_fixed_companders ${GOTO_LIB})


# Model Precision Tests
add_executable(model_prec_sgemm
	tests/model_prec_sgemm.cpp
  	${TEST_COMMON_SRC_FILES})
set_property(
  TARGET model_prec_sgemm
  APPEND PROPERTY COMPILE_DEFINITIONS EXTERNAL_A_QUANT
)
target_link_libraries(model_prec_sgemm gblas_external_aquant ${GOTO_LIB})


#add_executable(model_prec_sgemm_20110905
#        tests/model_prec_sgemm_20110905.cpp
#        ${TEST_COMMON_SRC_FILES})
#set_property(
#  TARGET model_prec_sgemm_20110905
#  APPEND PROPERTY COMPILE_DEFINITIONS EXTERNAL_A_QUANT
#)
#target_link_libraries(model_prec_sgemm_20110905 gblas_external_aquant ${GOTO_LIB})

add_executable(model_prec_dgemm
        tests/model_prec_dgemm.cpp
        ${TEST_COMMON_SRC_FILES})
set_property(
  TARGET model_prec_dgemm
  APPEND PROPERTY COMPILE_DEFINITIONS EXTERNAL_A_QUANT
)
target_link_libraries(model_prec_dgemm gblas_external_aquant ${GOTO_LIB})

add_executable(model_prec_dgemm_20110905
        tests/model_prec_dgemm_20110905.cpp
  	${TEST_COMMON_SRC_FILES})
set_property(
  TARGET model_prec_dgemm_20110905
  APPEND PROPERTY COMPILE_DEFINITIONS EXTERNAL_A_QUANT
)
target_link_libraries(model_prec_dgemm_20110905 gblas_external_aquant ${GOTO_LIB})

#pearson_coefficient
add_executable(pearson_coefficient
        tests/pearson_coefficient.cpp)
