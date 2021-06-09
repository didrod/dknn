#pragma once

#include "dknn/typedefs.hpp"

namespace dknn {
  /**
   * finds nearest k features distributedly using MPI.
   */
  std::vector<feature_id_set_t> mpi_brute_force_nearest_k(
    size_t k, feature_id_set_t const& train_feature_ids,
    feature_set_t const& query_set);
}  // namespace dknn
