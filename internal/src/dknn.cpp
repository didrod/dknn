#include "dknn/dknn.hpp"
#include "dknn/__detail__/dataset.hpp"
#include "dknn/__detail__/local_jobs.hpp"
#include "dknn/__detail__/gather.hpp"

#include <mpi.h>

#include <boost/format.hpp>
#include <string>

namespace dknn {
  using std::string;
  using std::vector;

  static int _rank = 0;
  static int _world_size = 1;

  bool init(int* pargc, char*** pargv) {
    MPI_Init(pargc, pargv);

    MPI_Comm_size(MPI_COMM_WORLD, &_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);

    if (_world_size <= 0) {
      auto fmt = boost::format("invalid world size: %1%");
      std::cerr << fmt % _world_size << std::endl;
      return false;
    }

    if (_rank < 0 || _rank >= _world_size) {
      auto fmt = boost::format("invalid node rank: %1%; world size = %2%");
      std::cerr << fmt % _rank % _world_size << std::endl;
      return false;
    }

    if (!__dknn_gather__init()) {
      auto fmt = boost::format("MPI type commit failed");
      std::cerr << fmt << std::endl;
      return false;
    }

    auto argv = *pargv;
    load_feature_dataset(argv[1]);

    return true;
  }

  void term() {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
  }

  int node_rank() {
    return _rank;
  }

  int world_size() {
    return _world_size;
  }

  static knn_set_query_result_t crop_nearest_k(
    size_t k, knn_set_query_result_t gathered_set_query_result) {
    knn_set_query_result_t result;

    for (auto& query_result : gathered_set_query_result) {
      auto match_comparison = [](auto const& a, auto const& b) {
        auto const& [id_a, distance_a, class_a] = a;
        auto const& [id_b, distance_b, class_b] = b;
        return distance_a <= distance_b;
      };
      std::partial_sort(
        query_result.begin(), query_result.begin() + k, query_result.end(),
        match_comparison);

      knn_query_result_t nearest_k;
      for (size_t i = 0; i < k; i++)
        nearest_k.emplace_back(query_result.at(i));
      result.emplace_back(nearest_k);
    }
    return result;
  }

  static feature_class_t classify(knn_query_result_t const& result) {
    std::map<feature_class_t, size_t> counts;
    // zero-initialize
    for (auto const& [_, __, _class] : result)
      counts[_class] = 0;

    for (auto const& [_, __, _class] : result)
      counts[_class]++;

    auto count_compare =
      [](auto const& class_count_a, auto const& class_count_b) {
        auto const& [class_a, count_a] = class_count_a;
        auto const& [class_b, count_b] = class_count_b;
        return count_a < count_b;
      };

    auto const& [max_class, max_count] =
      *std::max_element(counts.begin(), counts.end(), count_compare);
    return max_class;
  }

  static vector<feature_class_t> classify(
    knn_set_query_result_t const& set_query_results) {
    vector<feature_class_t> result;
    for (auto const& query_result : set_query_results)
      result.emplace_back(classify(query_result));
    return result;
  }

  vector<feature_class_t> mpi_brute_force_nearest_k(
    size_t k, feature_set_t const& query_set) {
    // this part is executed in *scattered context*
    // (i.e. executed in each node concurrently).
    auto scattered_knn_results = nearest_k(k, query_set);

    // then we gather the result to the master node.
    auto gathered_knn_results = gather(k, scattered_knn_results);
    auto nearest_k_set = crop_nearest_k(k, gathered_knn_results);
    return classify(nearest_k_set);
  }
}  // namespace dknn
