#include "dknn/__detail__/scatter.hpp"
#include "dknn/dknn.hpp"

#include <mpi.h>
#include <vector>

namespace dknn {
  using std::vector;

  static vector<double> flatten(feature_set_t const& query_set) {
    vector<double> result;
    for (auto const& feature : query_set)
      result.insert(result.end(), feature.begin(), feature.end());
    return result;
  }

  static feature_set_t unflatten(
    vector<double> const& flatten_query_set, size_t size, size_t dimension) {
    feature_set_t result;
    result.resize(size);

    for (size_t i = 0; i < size; i++) {
      auto s = flatten_query_set.data() + dimension * i;
      auto e = s + dimension;
      result[i].insert(result[i].end(), s, e);
    }
    return result;
  }

  void send_query(feature_set_t const& query_set) {
    if (node_rank() != 0)
      return;

    uint64_t size = query_set.size();
    MPI_Bcast(&size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (size == 0)
      return;

    uint64_t dimension = query_set.front().size();
    MPI_Bcast(&dimension, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (dimension == 0)
      return;

    auto flat_query_set = flatten(query_set);
    MPI_Bcast(
      flat_query_set.data(), size * dimension, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  }

  feature_set_t receive_query() {
    if (node_rank() == 0)
      return {};

    uint64_t size;
    MPI_Bcast(&size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (size == 0)
      return {};

    uint64_t dimension;
    MPI_Bcast(&dimension, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (dimension == 0)
      return {};

    vector<double> flat_query_set;
    flat_query_set.resize(size * dimension);
    MPI_Bcast(
      flat_query_set.data(), size * dimension, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    return unflatten(flat_query_set, size, dimension);
  }
}  // namespace dknn
