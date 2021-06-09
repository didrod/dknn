#include "dknn/dknn.hpp"
#include <omp.h>
#include <mpi.h>

namespace dknn {
  feature_id_set_t __mpi_brute_force_nearest_k__(
    mpi_feature_set_flatten_data_t const& flatten_train_data,
    mpi_feature_set_flatten_data_t const& flatten_query_data) {
    // TODO: implement this
    return {};
  }
}  // namespace dknn
