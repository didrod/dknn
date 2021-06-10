#include "dknn/typedefs.hpp"

namespace dknn {
  bool __dknn_gather__init();

  knn_set_query_result_t gather(
    size_t k, knn_set_query_result_t const& scattered_results);
}  // namespace dknn
