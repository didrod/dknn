#include "dknn/dknn.hpp"
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
    size_t k, knn_set_query_result_t const& gathered_knn_results) {
    // TODO: crop nearest K
    return {};
  }

  vector<feature_id_set_t> mpi_brute_force_nearest_k(
    size_t k, feature_id_set_t const& train_feature_ids,
    feature_set_t const& query_set) {
    auto subworker_train_features = scatter(train_feature_ids);

    // from here is executed in *scattered context*
    // (i.e. executed in each node concurrently).
    load_train_data(subworker_train_features);
    auto scattered_knn_results = nearest_k(k, query_set);

    // now we go back to *gathered context*.
    auto gathered_knn_results =
      gather(k, query_set.size(), scattered_knn_results);

    return crop_nearest_k(k, gathered_knn_results);
  }
}  // namespace dknn
