add_library(diff_drive_core INTERFACE)
target_include_directories(diff_drive_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
target_link_libraries(diff_drive_core INTERFACE Eigen3::Eigen)
target_compile_features(diff_drive_core INTERFACE cxx_std_17)

add_subdirectory(tests)
