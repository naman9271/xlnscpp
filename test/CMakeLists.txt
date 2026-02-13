# ============================================================================
# XLNSCPP — Test Suite
# ============================================================================

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

add_executable(test_xlns32 unit/test_xlns32.cpp)
target_link_libraries(test_xlns32 PRIVATE xlns::xlns)
add_test(NAME xlns32_arithmetic COMMAND test_xlns32)

add_executable(test_xlns16 unit/test_xlns16.cpp)
target_link_libraries(test_xlns16 PRIVATE xlns::xlns)
add_test(NAME xlns16_arithmetic COMMAND test_xlns16)

add_executable(test_xlns_both unit/test_xlns_both.cpp)
target_link_libraries(test_xlns_both PRIVATE xlns::xlns)
add_test(NAME xlns_coexistence COMMAND test_xlns_both)

# ---------------------------------------------------------------------------
# Interactive / function testers (not run in CI, but built)
# ---------------------------------------------------------------------------

add_executable(xlns32_funtest unit/xlns32_funtest.cpp)
target_link_libraries(xlns32_funtest PRIVATE xlns::xlns)

add_executable(xlns16_funtest unit/xlns16_funtest.cpp)
target_link_libraries(xlns16_funtest PRIVATE xlns::xlns)

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

if(XLNS_BUILD_BENCHMARKS)
    add_executable(bench_xlns16 benchmark/bench_xlns16.cpp)
    target_link_libraries(bench_xlns16 PRIVATE xlns::xlns)
endif()

# ---------------------------------------------------------------------------
# Gaussian log cross-validation CLI tools
#
# sb16 / db16   → need xlns16_ideal (exact sb/db via math.h)
# sbmit16/dbmit16 → must NOT use ideal (tests approximate Mitchell path)
#
# Because each tool requires different compile-time flags, they compile
# xlns16.cpp directly with the correct definitions rather than linking
# against the library target (which has a single, fixed flag set).
# ---------------------------------------------------------------------------

set(_XLNS16_SRC "${CMAKE_SOURCE_DIR}/src/xlns16.cpp")

# -- sb16 (ideal) --
add_executable(sb16 cross-validation/sb16.cpp ${_XLNS16_SRC})
target_include_directories(sb16 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(sb16 PRIVATE xlns16_ideal)
if(NOT MSVC)
    target_link_libraries(sb16 PRIVATE m)
endif()

# -- db16 (ideal) --
add_executable(db16 cross-validation/db16.cpp ${_XLNS16_SRC})
target_include_directories(db16 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(db16 PRIVATE xlns16_ideal)
if(NOT MSVC)
    target_link_libraries(db16 PRIVATE m)
endif()

# -- sbmit16 (approximate / Mitchell) --
add_executable(sbmit16 cross-validation/sbmit16.cpp ${_XLNS16_SRC})
target_include_directories(sbmit16 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
if(NOT MSVC)
    target_link_libraries(sbmit16 PRIVATE m)
endif()

# -- dbmit16 (approximate / Mitchell) --
add_executable(dbmit16 cross-validation/dbmit16.cpp ${_XLNS16_SRC})
target_include_directories(dbmit16 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
if(NOT MSVC)
    target_link_libraries(dbmit16 PRIVATE m)
endif()

# Copy Python cross-validation scripts next to binaries
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cross-validation/sbtest.py
    ${CMAKE_CURRENT_BINARY_DIR}/sbtest.py COPYONLY)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cross-validation/dbtest.py
    ${CMAKE_CURRENT_BINARY_DIR}/dbtest.py COPYONLY)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cross-validation/sblptest.py
    ${CMAKE_CURRENT_BINARY_DIR}/sblptest.py COPYONLY)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cross-validation/dblptest.py
    ${CMAKE_CURRENT_BINARY_DIR}/dblptest.py COPYONLY)
