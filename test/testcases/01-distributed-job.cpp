#include "dknn/dknn.hpp"
#include "dknn/__detail__/dataset.hpp"

#include <algorithm>
#include <random>
#include <catch2/catch.hpp>

static auto scatter(
  std::mt19937& rgen, dknn::feature_id_set_t const& ids, double x, double y) {
  std::normal_distribution<> rand(0, 0.1);

  dknn::id_feature_dict_t result;
  for (auto id : ids)
    result.emplace(id, dknn::feature_t {x + rand(rgen), y + rand(rgen)});
  return result;
}

TEST_CASE("Test distributed knn search", "[distributed-0]") {
  auto rank = dknn::node_rank();
  auto nworkers = dknn::world_size();

  std::mt19937 rgen(20210610 + 31 * rank);

  REQUIRE(nworkers == 4);
  dknn::__local_train_feature_cache__.clear();

  switch (rank) {
  case 0:
    dknn::__local_train_feature_cache__ = scatter(rgen, {1, 2, 3, 4}, 0, 0);
    break;
  case 1:
    dknn::__local_train_feature_cache__ = scatter(rgen, {5, 6, 7}, 1.0, 0);
    break;
  case 2:
    dknn::__local_train_feature_cache__ = scatter(rgen, {13, 15, 17}, 5.0, 5.0);
    break;
  case 3:
    dknn::__local_train_feature_cache__ =
      scatter(rgen, {18, 19, 20}, 9.0, 10.0);
    break;
  default:
    CAPTURE("node rank seems to be incorrect");
    CAPTURE(rank);
    REQUIRE(false);
    break;
  }

  auto query_results = dknn::mpi_brute_force_nearest_k(
    3, {1, 2, 3, 5, 6, 7, 13, 15, 17, 18, 19, 20},
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

  auto cluster_01 = dknn::feature_id_set_t {1, 2, 3, 5, 6, 7};

  CHECK(q0 == dknn::feature_id_set_t {1, 2, 3});
  CHECK(
    std::includes(cluster_01.begin(), cluster_01.end(), q1.begin(), q1.end()));
  CHECK(q2 == dknn::feature_id_set_t {13, 15, 17});
  CHECK(q3 == dknn::feature_id_set_t {18, 19, 20});

  CHECK(q4.size() == 3);
}
