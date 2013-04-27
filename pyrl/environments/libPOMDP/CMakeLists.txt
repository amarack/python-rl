cmake_minimum_required(VERSION 2.8)

PROJECT(libPOMDP)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

file(GLOB SRC
    "src/*.h"
    "src/*.c"
)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)

# Semi-hacky way of getting numpy included
# Needs testing on linux
STRING(REPLACE bin/python lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/numpy/core/include/numpy/ NUMPY_INC ${PYTHON_EXECUTABLE})

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INC})

BISON_TARGET(parser ${CMAKE_CURRENT_SOURCE_DIR}/src/parser.y ${CMAKE_CURRENT_SOURCE_DIR}/src/parser.c)
FLEX_TARGET(scanner ${CMAKE_CURRENT_SOURCE_DIR}/src/scanner.l  ${CMAKE_CURRENT_SOURCE_DIR}/src/scanner.c)
ADD_FLEX_BISON_DEPENDENCY(scanner parser)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(libpomdp MODULE ${SRC} ${BISON_parser_OUTPUTS} ${FLEX_scanner_OUTPUTS})
set_target_properties(libpomdp PROPERTIES PREFIX "")
target_link_libraries(libpomdp ${PYTHON_LIBRARIES} ${FLEX_LIBRARIES} ${BISON_LIBRARIES})
