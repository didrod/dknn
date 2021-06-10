#pragma once

#include "dknn/typedefs.hpp"

namespace dknn {
  knn_query_result_t nearest_k(size_t k, feature_t const& query_feature);
  knn_set_query_result_t nearest_k(size_t k, feature_set_t const& query_set);
}  // namespace dknn
