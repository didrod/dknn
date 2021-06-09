include(ProcessorCount)
ProcessorCount(JOBS)

configure_file(
  cmake/catch2.cmake.in
  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/catch2-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/catch2-download"
)
execute_process(COMMAND "${CMAKE_COMMAND}" --build . -- -j${JOBS}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/catch2-download"
)
find_package(Catch2 REQUIRED
  PATHS ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/catch2-install
  NO_DEFAULT_PATH
)
