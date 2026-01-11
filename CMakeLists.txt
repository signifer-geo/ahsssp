cmake_minimum_required(VERSION 3.16)
project(ahsssp
        VERSION 0.1.0
        DESCRIPTION "Adaptive Hierarchical Single-Source Shortest Path (AH-SSSP)"
        LANGUAGES CXX)

# ------------------------------------------------------------
# Build settings
# ------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------

option(AHSSSP_BUILD_TESTS "Build tests" ON)
option(AHSSSP_BUILD_CLI   "Build command line tool" ON)

# ------------------------------------------------------------
# Library
# ------------------------------------------------------------

add_library(ahsssp
    ahsssp.cpp
    ahsssp.hpp
)

target_include_directories(ahsssp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(ahsssp PUBLIC cxx_std_17)

# Reasonable warnings
if(MSVC)
    target_compile_options(ahsssp PRIVATE /W4)
else()
    target_compile_options(ahsssp PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ------------------------------------------------------------
# CLI executable
# ------------------------------------------------------------

if(AHSSSP_BUILD_CLI)
    add_executable(ahsssp_cli
        ahsssp.cpp
    )
    target_link_libraries(ahsssp_cli PRIVATE ahsssp)
endif()

# ------------------------------------------------------------
# Tests (optional, future-proof)
# ------------------------------------------------------------

if(AHSSSP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
