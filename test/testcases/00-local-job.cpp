#include "dknn/__detail__/local_jobs.hpp"
#include "dknn/__detail__/dataset.hpp"
#include <catch2/catch.hpp>

namespace dknn {
  static feature_id_set_t feature_ids(knn_query_result_t const& query_result) {
    feature_id_set_t result;
    for (auto const& [id, _] : query_result)
      result.emplace(id);
    return result;
  }

  TEST_CASE("Test node-local knn search", "[local-0]") {
    __feature_dataset__.clear();
    __feature_dataset__ = {
      // class 1
      {1, {0.0, 0.0}},
      {2, {0.1, 0.0}},
      {3, {0.0, 0.1}},

      // class 2
      {4, {1.0, 1.0}},
      {5, {0.9, 1.0}},
      {6, {1.0, 1.1}},
    };

    auto result = nearest_k(3, {{0.0, 0.0}, {1.0, 1.0}});
    REQUIRE(result.size() == 2);

    CHECK(feature_ids(result[0]) == feature_id_set_t {1, 2, 3});
    CHECK(feature_ids(result[1]) == feature_id_set_t {4, 5, 6});
  }
}  // namespace dknn
