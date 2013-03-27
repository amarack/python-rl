cmake_minimum_required(VERSION 2.8)

PROJECT(libPOMDP)

set(SRC scanner.c parser.c mdp-common.h mdp.h mdp.c parse_err.h parse_err.c imm-reward.h imm-reward.c libpomdp.h libpomdp.c parse_hash.h parse_hash.c sparse-matrix.h sparse-matrix.c)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)

# Semi-hacky way of getting numpy included
# Needs testing on linux
STRING(REPLACE bin/python lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/numpy/core/include/numpy/ NUMPY_INC ${PYTHON_EXECUTABLE})

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INC})

BISON_TARGET(parser parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.c)
FLEX_TARGET(scanner scanner.l  ${CMAKE_CURRENT_BINARY_DIR}/scanner.c)
ADD_FLEX_BISON_DEPENDENCY(scanner parser)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(libpomdp MODULE ${SRC} ${BISON_MyParser_OUTPUTS} ${FLEX_MyScanner_OUTPUTS})
set_target_properties(libpomdp PROPERTIES PREFIX "")
target_link_libraries(libpomdp ${PYTHON_LIBRARIES} ${FLEX_LIBRARIES} ${BISON_LIBRARIES})
