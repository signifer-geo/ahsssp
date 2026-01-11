cmake_minimum_required(VERSION 3.16)

project(ahsssp
  VERSION 0.1.0
  DESCRIPTION "Adaptive Hierarchical Single-Source Shortest Path (AH-SSSP)"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AHSSSP_BUILD_TESTS "Build tests" ON)

# --- Main executable (matches your current repo style: ahsssp.cpp -> ahsssp) ---
add_executable(ahsssp
  ahsssp.cpp
)

target_include_directories(ahsssp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
  target_compile_options(ahsssp PRIVATE /W4)
else()
  target_compile_options(ahsssp PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- Tests ---
if(AHSSSP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
