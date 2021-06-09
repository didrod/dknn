target_sources(
  dknn_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/00-local-knn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/01-distributed.cpp
)
