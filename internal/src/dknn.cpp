#include "dknn/dknn.hpp"
#include "dknn/local_knn.hpp"
#include <mpi.h>

namespace dknn {
  using std::vector;

  static vector<feature_id_t> scatter_train_feature_ids(
    int rank, int workers, feature_id_set_t const& train_feature_ids) {
    vector<feature_id_t> flatten_train_feature_ids;
    if (rank == 0) {
      flatten_train_feature_ids.reserve(train_feature_ids.size());
      flatten_train_feature_ids.insert(
        flatten_train_feature_ids.end(), train_feature_ids.begin(),
        train_feature_ids.end());
    }

    vector<feature_id_t> subworker_train_feature_ids;
    int subworker_trainset_size = train_feature_ids.size() / workers;
    if (subworker_trainset_size * workers != train_feature_ids.size()) {
      std::cerr << "train feature size should be multiple of workers"
                << std::endl;
      return {};
    }

    subworker_train_feature_ids.resize(subworker_trainset_size);

    MPI_Scatter(
      flatten_train_feature_ids.data(), subworker_trainset_size, MPI_UINT64_T,
      subworker_train_feature_ids.data(), subworker_trainset_size, MPI_UINT64_T,
      0, MPI_COMM_WORLD);
    return subworker_train_feature_ids;
  }

  static vector<feature_id_set_t> gather_local_knn_results(
    int rank, int workers, size_t k, size_t query_set_size,
    vector<feature_id_set_t> const& local_knn_results) {
    if (query_set_size != local_knn_results.size()) {
      std::cerr << "query set size and local knn result size mismatches!"
                << std::endl;
      return {};
    }

    vector<feature_id_t> local_flatten_knn_results;
    local_flatten_knn_results.reserve(k * query_set_size);
    for (auto const& knn_result : local_knn_results) {
      local_flatten_knn_results.insert(
        local_flatten_knn_results.end(), knn_result.begin(), knn_result.end());
    }

    size_t gather_size = workers * k * query_set_size;
    vector<feature_id_t> gathered_flat_knn_results;
    if (rank == 0)
      gathered_flat_knn_results.resize(gather_size);
    MPI_Gather(
      local_flatten_knn_results.data(), k * query_set_size, MPI_UINT64_T,
      gathered_flat_knn_results.data(), gather_size, MPI_UINT64_T, 0,
      MPI_COMM_WORLD);

    vector<feature_id_set_t> gathered_knn_results;
    gathered_knn_results.resize(query_set_size);
    for (int r = 0; r < workers; r++) {
      for (size_t i = 0; i < query_set_size; i++) {
        auto& knn_result = gathered_knn_results[i];

        auto data_start =
          &gathered_flat_knn_results[r * k * query_set_size + k * i];
        auto data_end = data_start + k;
        knn_result.insert(data_start, data_end);
      }
    }
    return gathered_knn_results;
  }

  static vector<feature_id_set_t> crop_nearest_k(
    size_t k, vector<feature_id_set_t> const& gathered_knn_results) {
    // TODO: crop nearest K
    return {};
  }

  vector<feature_id_set_t> mpi_brute_force_nearest_k(
    size_t k, feature_id_set_t const& train_feature_ids,
    feature_set_t const& query_set) {
    int workers = -1, rank = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &workers);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (workers <= 0) {
      std::cerr << "invalid world size: " << workers << std::endl;
      return {};
    }

    if (rank < 0 || rank >= workers) {
      std::cerr << "invalid process rank: " << rank << std::endl;
      return {};
    }

    // from here is executed in *scattered context*
    // (i.e. executed in each node concurrently).
    auto subworker_train_features =
      scatter_train_feature_ids(rank, workers, train_feature_ids);
    load_cache(subworker_train_features);
    auto local_knn_results = node_local_nearest_k(k, query_set);
    auto gathered_knn_results = gather_local_knn_results(
      rank, workers, k, query_set.size(), local_knn_results);
    // now we go back to *gathered context*.

    return crop_nearest_k(k, gathered_knn_results);
  }
}  // namespace dknn
