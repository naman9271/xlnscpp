# ============================================================================
# XLNSCPP — Logarithmic Number System Library for C++
# Copyright (c) 1999–2025 Mark G. Arnold
# SPDX-License-Identifier: MIT
# ============================================================================

cmake_minimum_required(VERSION 3.16)

project(xlns
    VERSION 1.0.0
    DESCRIPTION "C++ Logarithmic Number System (LNS) library — 16-bit and 32-bit"
    HOMEPAGE_URL "https://github.com/xlnsresearch/xlnscpp"
    LANGUAGES CXX
)

# ============================================================================
# Build options
# ============================================================================

option(XLNS_BUILD_TESTS      "Build the test suite"                ON)
option(XLNS_BUILD_BENCHMARKS "Build performance benchmarks"        ON)
option(XLNS_BUILD_EXAMPLES   "Build example programs"              ON)
option(XLNS_INSTALL          "Generate install targets"            ON)

# Library configuration options (compile-time switches)
option(XLNS32_IDEAL   "Use math.h for 32-bit sb/db (most accurate)"     OFF)
option(XLNS32_ALT     "Use alternative 32-bit addition algorithm"        OFF)
option(XLNS16_IDEAL   "Use math.h for 16-bit sb/db (most accurate)"     OFF)
option(XLNS16_ALT     "Use alternative 16-bit addition algorithm"        ON)
option(XLNS16_ALTOPT  "Use simplified LPVIP within alt (less accurate)"  OFF)
option(XLNS16_TABLE   "Use lookup tables for 16-bit conversions & sb/db" OFF)

# ============================================================================
# Global settings
# ============================================================================

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ============================================================================
# Library target
# ============================================================================

add_library(xlns STATIC
    src/xlns32.cpp
    src/xlns16.cpp
)

# Public headers — consumers see include/xlns/*.h
target_include_directories(xlns
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link math library on non-MSVC platforms
if(NOT MSVC)
    target_link_libraries(xlns PUBLIC m)
endif()

# Apply compile-time configuration options
if(XLNS32_IDEAL)
    target_compile_definitions(xlns PUBLIC xlns32_ideal)
endif()
if(XLNS32_ALT)
    target_compile_definitions(xlns PUBLIC xlns32_alt)
endif()
if(XLNS16_IDEAL)
    target_compile_definitions(xlns PUBLIC xlns16_ideal)
endif()
if(XLNS16_ALT)
    target_compile_definitions(xlns PUBLIC xlns16_alt)
endif()
if(XLNS16_ALTOPT)
    target_compile_definitions(xlns PUBLIC xlns16_altopt)
endif()
if(XLNS16_TABLE)
    target_compile_definitions(xlns PUBLIC xlns16_table)
endif()

# Set library version properties
set_target_properties(xlns PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
)

# Create namespaced alias for use in subdirectories and as a subproject
add_library(xlns::xlns ALIAS xlns)

# ============================================================================
# Tests
# ============================================================================

if(XLNS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# ============================================================================
# Examples
# ============================================================================

if(XLNS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Install rules
# ============================================================================

if(XLNS_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Install the library binary
    install(TARGETS xlns
        EXPORT xlnsTargets
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    # Install public headers
    install(DIRECTORY include/xlns/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xlns
        FILES_MATCHING PATTERN "*.h"
    )

    # Generate and install the export set
    install(EXPORT xlnsTargets
        FILE xlnsTargets.cmake
        NAMESPACE xlns::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xlns
    )

    # Generate the package config file
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/xlnsConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/xlnsConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xlns
    )

    # Generate the version file
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/xlnsConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    # Install config and version files
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/xlnsConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/xlnsConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xlns
    )
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== XLNSCPP Configuration ===")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  32-bit ideal:     ${XLNS32_IDEAL}")
message(STATUS "  32-bit alt:       ${XLNS32_ALT}")
message(STATUS "  16-bit ideal:     ${XLNS16_IDEAL}")
message(STATUS "  16-bit alt:       ${XLNS16_ALT}")
message(STATUS "  16-bit altopt:    ${XLNS16_ALTOPT}")
message(STATUS "  16-bit table:     ${XLNS16_TABLE}")
message(STATUS "  Build tests:      ${XLNS_BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${XLNS_BUILD_BENCHMARKS}")
message(STATUS "  Build examples:   ${XLNS_BUILD_EXAMPLES}")
message(STATUS "=============================")
message(STATUS "")
