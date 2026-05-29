cmake_minimum_required(VERSION 3.18)
project(lattice_dsp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

pybind11_add_module(_core MODULE
  src/bindings.cpp
  src/lattice.cpp
  src/model_reduction.cpp
)

target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(_core PRIVATE cxx_std_17)

if(OpenMP_CXX_FOUND)
  target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(_core PRIVATE LATTICE_DSP_HAS_OPENMP=1)
endif()

install(TARGETS _core LIBRARY DESTINATION lattice_dsp)
