cmake_minimum_required(VERSION 3.6)
project(INtrinsics)

################################
# GTest
################################
ADD_SUBDIRECTORY (${CMAKE_SOURCE_DIR}/googletest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

################################
# Unit Tests
################################

# These are compiler flags - these should be used
# to give you a view of what the final code will look like
add_definitions("-Ofast -mavx2")

# There's many flags you can enable for warnings.
# This, however, seems to be a decent subset
# The fact that -Weverything is so insane really is the biggest problem of our time.
add_definitions("-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Winit-self -Wlogical-op -Wmissing-declarations -Wnoexcept -Woverloaded-virtual -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wno-unused -Wdisabled-optimization")
# Add test cpp files
add_executable(runIntrinsicsTests intrinsics.t.cpp)
# Link test executable against gtest & gtest_main
target_link_libraries(runIntrinsicsTests gtest gtest_main pthread)

# Add here if you want ctest to pick up these tests.
# There's no reason not to have all tests here, as it makes
# running CI as simple as running Ctest
add_test(runIntrinsicsTests runIntrinsicsTests)
