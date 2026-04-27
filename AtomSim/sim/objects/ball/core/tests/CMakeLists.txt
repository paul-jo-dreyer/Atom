add_executable(ball_tests
    test_stub.cpp
)
target_link_libraries(ball_tests PRIVATE
    ball_core
    doctest::doctest
)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(ball_tests)
