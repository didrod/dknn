project(dknn)
cmake_minimum_required(VERSION 3.10)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Boost REQUIRED)

add_library(internal "")
add_subdirectory(internal)
target_include_directories(
  internal PUBLIC internal/include)
target_link_libraries(
  internal PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX Boost::boost)
target_compile_features(internal PUBLIC cxx_std_17)
target_compile_options(
  internal PRIVATE
  -ffast-math
  -O2
  -g3
)

add_executable(main "")
add_subdirectory(src)
target_link_libraries(main PRIVATE internal MPI::MPI_CXX)
target_compile_features(main PRIVATE cxx_std_17)

include(cmake/configure_catch2.cmake)
add_executable(dknn_test "")
add_subdirectory(test)
target_link_libraries(
  dknn_test PRIVATE internal MPI::MPI_CXX Catch2::Catch2)
target_compile_features(dknn_test PRIVATE cxx_std_17)

add_custom_target(run DEPENDS main)
add_custom_command(
  TARGET run
  COMMAND mpiexec -n 4 ${CMAKE_CURRENT_BINARY_DIR}/main
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  POST_BUILD
)

add_custom_target(run_test DEPENDS dknn_test)
add_custom_command(
  TARGET run_test
  COMMAND mpiexec -n 4 ${CMAKE_CURRENT_BINARY_DIR}/dknn_test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  POST_BUILD
)
