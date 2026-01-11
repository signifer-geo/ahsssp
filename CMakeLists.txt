cmake_minimum_required(VERSION 3.16)

project(ahsssp
  VERSION 0.1.0
  DESCRIPTION "AH-SSSP: Fast Single-Source Shortest Paths"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AHSSSP_BUILD_TESTS  "Build tests" ON)
option(AHSSSP_BUILD_SIMPLE "Build the stable/simple engine" ON)
option(AHSSSP_BUILD_FULL   "Build the research/full engine" OFF)

# Warnings helper
function(ahsssp_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

# --- Simple engine (recommended) ---
if(AHSSSP_BUILD_SIMPLE)
  add_executable(ahsssp_simple
    src/simple/ahsssp_simple.cpp
  )
  target_include_directories(ahsssp_simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  ahsssp_set_warnings(ahsssp_simple)
endif()

# --- Full engine (research) ---
if(AHSSSP_BUILD_FULL)
  add_executable(ahsssp_full
    src/full/ahsssp.cpp
  )
  target_include_directories(ahsssp_full PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/full)
  ahsssp_set_warnings(ahsssp_full)
endif()

# --- Tests ---
if(AHSSSP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

