#pragma once

#include "dknn/typedefs.hpp"
#include "dknn/__detail__/scatter.hpp"

namespace dknn {
  bool init(int* pargc, char*** pargv);
  void term();

  int node_rank();
  int world_size();

  /**
   * finds nearest k features distributedly using MPI.
   */
  std::vector<feature_class_t> mpi_brute_force_nearest_k(
    size_t k, feature_set_t const& query_set);

  template <typename query_loader_t>
  std::vector<feature_class_t> mpi_brute_force_nearest_k(
    size_t k, query_loader_t query_loader) {
    if (node_rank() == 0) {
      auto query_set = query_loader();
      send_query(query_set);
      return mpi_brute_force_nearest_k(k, query_set);
    } else {
      auto query_set = receive_query();
      mpi_brute_force_nearest_k(k, query_set);
      return {};
    }
  }
}  // namespace dknn
