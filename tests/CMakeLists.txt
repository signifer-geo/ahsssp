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

