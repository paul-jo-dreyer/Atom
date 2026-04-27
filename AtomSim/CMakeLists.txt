cmake_minimum_required(VERSION 3.20)
project(AtomSim LANGUAGES CXX)

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

enable_testing()

# Subprojects
add_subdirectory(vehicle_dynamics/diff_drive/core)
add_subdirectory(vehicle_dynamics/diff_drive/bindings)
add_subdirectory(sim/objects/ball/core)
add_subdirectory(sim/objects/ball/bindings)
# add_subdirectory(sim)              # later, M4
# add_subdirectory(tests)            # later, top-level integration tests