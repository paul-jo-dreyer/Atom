add_executable(sim_tests
    test_world.cpp
    test_robot.cpp
    test_ball.cpp
)
target_link_libraries(sim_tests PRIVATE
    sim_lib
    doctest::doctest
)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(sim_tests)
