cmake_minimum_required(VERSION 3.14)
project(GoTorch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_SEARCH_RESULT
    )
    if(pybind11_SEARCH_RESULT EQUAL 0 AND EXISTS "${pybind11_DIR}")
        find_package(pybind11 REQUIRED PATHS "${pybind11_DIR}")
    endif()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/GoTorch/backend/cppsrc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/GoTorch/backend/cppsrc
)

set(GOTORCH_SRCS
    GoTorch/backend/cppsrc/src/shape_utils.cpp
    GoTorch/backend/cppsrc/src/ndarray.cpp
    GoTorch/backend/cppsrc/src/kernels.cpp
)

if(TARGET pybind11::module)
    pybind11_add_module(native_backend
        ${GOTORCH_SRCS}
        GoTorch/backend/cppsrc/bindings.cpp
    )

    set_target_properties(native_backend PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/GoTorch/backend"
    )
endif()
