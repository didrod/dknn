#pragma once

#include <vector>
#include <tuple>
#include <map>
#include <set>

namespace dknn {
  using feature_id_t = size_t;
  using feature_index_t = size_t;
  using feature_id_set_t = std::set<feature_id_t>;

  struct mpi_feature_set_flatten_data_t {
    size_t set_size;
    size_t feature_dimension;
    std::vector<double> data;
    std::map<feature_index_t, feature_id_t> index_id_lookup;
  };

  /**
   * finds nearest k features distributedly using MPI.
   */
  feature_id_set_t __mpi_brute_force_nearest_k__(
    mpi_feature_set_flatten_data_t const& flatten_train_data,
    mpi_feature_set_flatten_data_t const& flatten_query_data);

  template <int dimension>
  using feature_t = std::array<double, dimension>;

  template <int dimension>
  using feature_set_t = std::map<feature_id_t, feature_t<dimension>>;

  /**
   * provides high-level interface to __mpi_brute_force_nearest_k__().
   */
  template <int dimension>
  feature_id_set_t brute_force_nearest_k(
    feature_set_t<dimension> const& train_data,
    feature_set_t<dimension> const& query_data) {
    // TODO
    return {};
  }
}  // namespace dknn
