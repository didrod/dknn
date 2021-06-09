#include "dknn/dknn.hpp"
#include <omp.h>
#include <mpi.h>

namespace dknn {
  static feature_set_t __local_train_feature_cache__ = {};

  static void load_cache(vector<feature_id_t> const& ids_to_load) {
    // TODO
  }

  feature_id_set_t mpi_brute_force_nearest_k(
    size_t k, feature_id_set_t const& train_feature_ids,
    feature_set_t const& query_set) {
    // TODO: implement this
    return {};
  }
}  // namespace dknn
