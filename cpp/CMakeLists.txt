cmake_minimum_required(VERSION 3.18)
project(framehands_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The Python modules are the reference: no fused multiply-adds (aarch64 contracts by default)
# and no reassociation, so results follow the documented evaluation order.
add_compile_options(-ffp-contract=off -fno-fast-math -Wall -Wextra)

add_library(framehands_core STATIC src/core.cpp)
target_include_directories(framehands_core PUBLIC include)

option(FRAMEHANDS_PYTHON "Build the framehands._core Python module" ON)
if(FRAMEHANDS_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core python/module.cpp)
  target_link_libraries(_core PRIVATE framehands_core)
endif()
