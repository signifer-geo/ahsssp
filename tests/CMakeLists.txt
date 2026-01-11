
add_executable(ahsssp_header_compiles
  header_compiles.cpp
)

target_include_directories(ahsssp_header_compiles PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_test(NAME header_compiles COMMAND ahsssp_header_compiles)

# Smoke test: build the main executable and run it.
# If your program requires args, switch to: COMMAND ahsssp --help
add_test(NAME ahsssp_smoke_run COMMAND ahsssp)
