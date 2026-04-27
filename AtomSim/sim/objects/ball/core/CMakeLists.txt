add_library(ball_core INTERFACE)
target_include_directories(ball_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
target_link_libraries(ball_core INTERFACE Eigen3::Eigen)
target_compile_features(ball_core INTERFACE cxx_std_17)

add_subdirectory(tests)
