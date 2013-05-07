cmake_minimum_required(VERSION 2.8)

PROJECT(mdptetris)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

file(GLOB SRC
    "src/*.h"
    "src/*.c"
    "src/*.C"
)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

add_library(mdptetris MODULE ${SRC})
set_target_properties(mdptetris PROPERTIES PREFIX "")
target_link_libraries(mdptetris ${PYTHON_LIBRARIES})
