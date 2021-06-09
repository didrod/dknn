#pragma once

#include "dknn/typedefs.hpp"

namespace dknn {
  void load_cache(std::vector<feature_id_t> const& ids_to_load);
  std::vector<feature_id_set_t> node_local_nearest_k(
    size_t k, feature_set_t const& query_set);
}  // namespace dknn
