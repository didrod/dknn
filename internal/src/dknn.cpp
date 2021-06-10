#include "dknn/dknn.hpp"
#include "dknn/__detail__/dataset.hpp"
#include "dknn/__detail__/local_jobs.hpp"
#include "dknn/__detail__/scatter_gather.hpp"

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

    if (!scatter_gather_init()) {
      auto fmt = boost::format("MPI scatter/gather type commit failed");
      std::cerr << fmt << std::endl;
      return false;
    }

    load_feature_dataset();

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

  static vector<feature_id_set_t> crop_nearest_k(
    size_t k, knn_set_query_result_t gathered_set_query_result) {
    vector<feature_id_set_t> result;

    for (auto& query_result : gathered_set_query_result) {
      auto match_comparison = [](auto const& a, auto const& b) {
        auto const& [id_a, distance_a, class_a] = a;
        auto const& [id_b, distance_b, class_b] = b;
        return distance_a <= distance_b;
      };
      std::partial_sort(
        query_result.begin(), query_result.begin() + k, query_result.end(),
        match_comparison);

      feature_id_set_t nearest_k_set;
      for (size_t i = 0; i < k; i++) {
        auto const& [id, distance, _class] = query_result.at(i);
        nearest_k_set.emplace(id);
      }
      result.emplace_back(nearest_k_set);
    }
    return result;
  }

  vector<feature_id_set_t> mpi_brute_force_nearest_k(
    size_t k, feature_id_set_t const& train_feature_ids,
    feature_set_t const& query_set) {
    // this part is executed in *scattered context*
    // (i.e. executed in each node concurrently).
    auto scattered_knn_results = nearest_k(k, query_set);

    // then we gather the results to the master node.
    auto gathered_knn_results =
      gather(k, query_set.size(), scattered_knn_results);

    return crop_nearest_k(k, gathered_knn_results);
  }
}  // namespace dknn
