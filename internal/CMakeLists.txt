target_sources(
  internal PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dknn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/__detail__/dataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/__detail__/local_jobs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/__detail__/gather.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/__detail__/scatter.cpp
)
