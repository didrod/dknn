#pragma once

#include "dknn/typedefs.hpp"
#include <boost/container/flat_map.hpp>
#include <tuple>
#include <map>

namespace dknn {
  using boost::container::flat_map;
  using id_feature_dict_t = flat_map<feature_id_t, feature_t>;
  extern id_feature_dict_t __local_train_feature_cache__;

  void load_train_data(std::vector<feature_id_t> const& ids_to_load);

  using feature_distance_t = double;
  using feature_class_t = uint64_t;
  using feature_match_info_t = std::tuple<feature_distance_t, feature_class_t>;

  using knn_query_result_t = std::map<feature_id_t, feature_match_info_t>;
  using knn_set_query_result_t = std::vector<knn_query_result_t>;

  knn_query_result_t nearest_k(size_t k, feature_t const& query_feature);
  knn_set_query_result_t nearest_k(size_t k, feature_set_t const& query_set);
}  // namespace dknn
