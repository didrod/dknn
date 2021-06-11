#include "dknn/dknn.hpp"
#include "dknn/__detail__/dataset.hpp"

#include <algorithm>
#include <random>
#include <catch2/catch.hpp>

static auto generate_feature_dataset(
  std::mt19937& rgen, dknn::feature_id_set_t const& ids, int cls, double x,
  double y) {
  std::normal_distribution<> rand(0, 0.1);

  dknn::feature_dataset_t result;
  for (auto id : ids) {
    auto feature = dknn::feature_t {x + rand(rgen), y + rand(rgen)};
    result.emplace(id, std::make_tuple(cls, feature));
  }
  return result;
}

static bool load_test_dataset(int rank) {
  std::mt19937 rgen(20210610 + 31 * rank);
  dknn::__feature_dataset__.clear();

  switch (rank) {
  case 0:
    dknn::__feature_dataset__ =
      generate_feature_dataset(rgen, {1, 2, 3}, 0, 0, 0);
    return true;
  case 1:
    dknn::__feature_dataset__ =
      generate_feature_dataset(rgen, {5, 6, 7}, 1, 1.0, 0);
    return true;
  case 2:
    dknn::__feature_dataset__ =
      generate_feature_dataset(rgen, {13, 15, 17}, 2, 5.0, 5.0);
    return true;
  case 3:
    dknn::__feature_dataset__ =
      generate_feature_dataset(rgen, {18, 19, 20}, 3, 9.0, 10.0);
    return true;
  default:
    return false;
  }
}

TEST_CASE("Test distributed knn search", "[distributed-0]") {
  auto rank = dknn::node_rank();
  auto nworkers = dknn::world_size();

  REQUIRE(nworkers == 4);

  if (!load_test_dataset(rank)) {
    CAPTURE("node rank seems to be incorrect");
    CAPTURE(rank);
    REQUIRE(false);
  }

  auto query_results = dknn::mpi_brute_force_nearest_k(
    3,
    {
      {0.0, 0.0},  // query #0: cluster 0
      {0.5, 0.0},  // query #1: midpoint of cluster 0 and cluster 1
      {5.0, 5.0},  // query #2: cluster 2
      {8.0, 8.0},  // query #3: slightly off cluster 3
      {1234567.0, -1234567.0},  // query #4: nowhere
    });
  if (rank != 0) {
    return;
  }

  REQUIRE(query_results.size() == 5);
  auto const& q0 = query_results.at(0);
  auto const& q1 = query_results.at(1);
  auto const& q2 = query_results.at(2);
  auto const& q3 = query_results.at(3);
  auto const& q4 = query_results.at(4);

  CHECK(q0 == 0);
  CHECK((q1 == 0 || q1 == 1));
  CHECK(q2 == 2);
  CHECK(q3 == 3);

  CHECK((q4 >= 0 && q4 < 4));
}

TEST_CASE(
  "Test distributed knn search from lazy loaded query set", "[distributed-0]") {
  auto rank = dknn::node_rank();
  auto nworkers = dknn::world_size();

  REQUIRE(nworkers == 4);

  if (!load_test_dataset(rank)) {
    CAPTURE("node rank seems to be incorrect");
    CAPTURE(rank);
    REQUIRE(false);
  }

  auto query_results = dknn::mpi_brute_force_nearest_k(3, []() {
    return dknn::feature_set_t {
      {0.0, 0.0},  // query #0: cluster 0
      {0.5, 0.0},  // query #1: midpoint of cluster 0 and cluster 1
      {5.0, 5.0},  // query #2: cluster 2
      {8.0, 8.0},  // query #3: slightly off cluster 3
      {1234567.0, -1234567.0},  // query #4: nowhere
    };
  });
  if (rank != 0) {
    return;
  }

  REQUIRE(query_results.size() == 5);
  auto const& q0 = query_results.at(0);
  auto const& q1 = query_results.at(1);
  auto const& q2 = query_results.at(2);
  auto const& q3 = query_results.at(3);
  auto const& q4 = query_results.at(4);

  CHECK(q0 == 0);
  CHECK((q1 == 0 || q1 == 1));
  CHECK(q2 == 2);
  CHECK(q3 == 3);

  CHECK((q4 >= 0 && q4 < 4));
}
