#pragma once

#include <vector>
#include <set>

namespace dknn {
  using feature_id_t = uint64_t;
  using feature_index_t = uint64_t;
  using feature_id_set_t = std::set<feature_id_t>;

  using feature_t = std::vector<double>;
  using feature_set_t = std::vector<feature_t>;
}  // namespace dknn
