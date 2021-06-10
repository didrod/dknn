#include "dknn/typedefs.hpp"

namespace dknn {
  bool scatter_gather_init();

  knn_set_query_result_t gather(
    size_t k, size_t query_set_size,
    knn_set_query_result_t const& scattered_results);
}  // namespace dknn
