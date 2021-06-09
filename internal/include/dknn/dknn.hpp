#pragma once

#include <vector>
#include <tuple>
#include <map>
#include <set>

namespace dknn {
  using feature_id_t = uint64_t;
  using feature_index_t = uint64_t;
  using feature_id_set_t = std::set<feature_id_t>;

  using feature_t = std::vector<double>;
  using feature_set_t = std::vector<feature_t>;

  /**
   * finds nearest k features distributedly using MPI.
   */
  feature_id_set_t mpi_brute_force_nearest_k(
    size_t k, feature_id_set_t const& train_feature_ids,
    feature_set_t const& query_set);
}  // namespace dknn
