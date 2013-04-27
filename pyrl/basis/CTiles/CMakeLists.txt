cmake_minimum_required(VERSION 2.8)

PROJECT(CTiles)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

file(GLOB SRC
    "src/*.h"
    "src/*.cpp"
    "src/*.C"
)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})


add_library(tiles MODULE ${SRC})
set_target_properties(tiles PROPERTIES PREFIX "")
target_link_libraries(tiles ${PYTHON_LIBRARIES})
