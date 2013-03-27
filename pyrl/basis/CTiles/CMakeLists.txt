cmake_minimum_required(VERSION 2.8)

PROJECT(CTiles)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

add_library(tiles MODULE tilesInt.C tiles.cpp)
set_target_properties(tiles PROPERTIES PREFIX "")
target_link_libraries(tiles ${PYTHON_LIBRARIES})
