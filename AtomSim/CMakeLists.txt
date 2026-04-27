cmake_minimum_required(VERSION 3.20)
project(AtomSim LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clangd

# Speed up rebuilds
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# Use mold if available
find_program(MOLD_LINKER mold)
if(MOLD_LINKER)
    add_link_options(-fuse-ld=mold)
endif()

# Warnings
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# Eigen
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# doctest (fetched, single header)
include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)

# Box2D 3.x (C rewrite). Disable samples/benchmarks/docs — we just want the lib.
set(BOX2D_SAMPLES    OFF CACHE BOOL "" FORCE)
set(BOX2D_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BOX2D_DOCS       OFF CACHE BOOL "" FORCE)
set(BOX2D_UNIT_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    box2d
    GIT_REPOSITORY https://github.com/erincatto/box2d.git
    GIT_TAG v3.1.0
)
FetchContent_MakeAvailable(box2d)
# Box2D's own CMake adds -Werror; GCC -O3 trips a false-positive
# -Wmaybe-uninitialized inside their math_functions header in release.
# Disable that one warning class for the box2d target.
if(TARGET box2d)
    target_compile_options(box2d PRIVATE -Wno-error=maybe-uninitialized)
endif()

enable_testing()

# Subprojects
add_subdirectory(vehicle_dynamics/diff_drive/core)
add_subdirectory(vehicle_dynamics/diff_drive/bindings)
add_subdirectory(sim/objects/ball/core)
add_subdirectory(sim/objects/ball/bindings)
add_subdirectory(sim)
add_subdirectory(sim/bindings)
# add_subdirectory(tests)            # later, top-level integration tests