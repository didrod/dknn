#include "dknn/typedefs.hpp"

namespace dknn {
  bool scatter_gather_init();

  std::vector<feature_id_t> scatter(feature_id_set_t const& train_feature_ids);
  knn_set_query_result_t gather(
    size_t k, size_t query_set_size,
    knn_set_query_result_t const& scattered_results);
}  // namespace dknn
