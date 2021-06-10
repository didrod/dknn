#pragma once

#include "dknn/typedefs.hpp"

namespace dknn {
  bool init(int* pargc, char*** pargv);
  void term();

  int node_rank();
  int world_size();

  /**
   * finds nearest k features distributedly using MPI.
   */
  std::vector<feature_id_set_t> mpi_brute_force_nearest_k(
    size_t k, feature_set_t const& query_set);
}  // namespace dknn
