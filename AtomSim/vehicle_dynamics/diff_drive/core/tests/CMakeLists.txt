add_executable(diff_drive_tests
    test_stub.cpp
)
target_link_libraries(diff_drive_tests PRIVATE
    diff_drive_core
    doctest::doctest
)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(diff_drive_tests)
