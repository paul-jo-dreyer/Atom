add_library(sim_lib
    world.cpp
    robot.cpp
    ball.cpp
)
# Built with -fPIC so it can be linked into the pybind11 module (a shared lib).
set_target_properties(sim_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(sim_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}        # for unprefixed includes inside sim/
    ${CMAKE_CURRENT_SOURCE_DIR}/..     # for `#include "sim/world.hpp"` from outside
)
target_link_libraries(sim_lib PUBLIC
    box2d
    diff_drive_core    # exposes "diff_drive/core/<header>" include path
    ball_core          # exposes "ball/core/<header>" include path
)
target_compile_features(sim_lib PUBLIC cxx_std_17)

add_subdirectory(tests)
