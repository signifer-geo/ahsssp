add_executable(ahsssp_header_compiles
  header_compiles.cpp
)

# Allow including headers from src/full if you include ahsssp.hpp there.
target_include_directories(ahsssp_header_compiles PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/full
)

add_test(NAME header_compiles COMMAND ahsssp_header_compiles)

# Smoke-run the stable engine if it's built
if(TARGET ahsssp_simple)
  add_test(NAME ahsssp_simple_smoke_run COMMAND ahsssp_simple)
endif()

add_executable(ahsssp_full_smoke
  ahsssp_full_smoke.cpp
)

target_include_directories(ahsssp_full_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/full
)

add_test(NAME ahsssp_full_smoke_run COMMAND ahsssp_full_smoke)


find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(
  NAME ahsssp_full_differential_py
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/test_differential.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

