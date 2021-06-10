#pragma once

#include <vector>
#include <set>
#include <map>

namespace dknn {
  using feature_id_t = uint64_t;
  using feature_index_t = uint64_t;
  using feature_id_set_t = std::set<feature_id_t>;

  using feature_t = std::vector<double>;
  using feature_set_t = std::vector<feature_t>;

  using feature_distance_t = double;
  using feature_class_t = uint64_t;
  using feature_match_info_t = std::tuple<feature_distance_t, feature_class_t>;

  using knn_query_result_t = std::map<feature_id_t, feature_match_info_t>;
  using knn_set_query_result_t = std::vector<knn_query_result_t>;
}  // namespace dknn
