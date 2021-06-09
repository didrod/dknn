#pragma once

#include "dknn/typedefs.hpp"
#include <boost/container/flat_map.hpp>

namespace dknn {
  using boost::container::flat_map;
  using id_feature_dict_t = flat_map<feature_id_t, feature_t>;
  extern id_feature_dict_t __local_train_feature_cache__;

  void load_train_data(std::vector<feature_id_t> const& ids_to_load);

  feature_id_set_t node_local_nearest_k(
    size_t k, feature_t const& query_feature);
  std::vector<feature_id_set_t> node_local_nearest_k(
    size_t k, feature_set_t const& query_set);
}  // namespace dknn
