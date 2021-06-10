target_sources(
  dknn_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/00-local-job.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/01-distributed-job.cpp
)
