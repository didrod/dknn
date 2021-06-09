#include "dknn/local_knn.hpp"

#include <boost/container/flat_map.hpp>
#include <omp.h>

namespace dknn {
  id_feature_dict_t __local_train_feature_cache__ = {};

  void load_train_data(std::vector<feature_id_t> const& ids_to_load) {
    // TODO
  }

  std::vector<feature_id_set_t> node_local_nearest_k(
    size_t k, feature_set_t const& query_set) {
    // TODO
    return {};
  }
}  // namespace dknn
