#include "dknn/dknn.hpp"
#include <omp.h>
#include <mpi.h>

#include <memory>
#include <string>
#include <iostream>
#include <iterator>

namespace dknn {
  using std::vector;

  static feature_set_t __local_train_feature_cache__ = {};

  static void load_cache(vector<feature_id_t> const& ids_to_load) {
    // TODO
  }

  static feature_id_set_t local_brute_force_nearest_k(
    size_t k, feature_set_t const& query_set) {
    // TODO
    return {};
  }

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

  static feature_id_set_t gather_nearest_k(
    int rank, int workers, size_t k, feature_id_set_t const& local_nearest_k) {
    vector<feature_id_t> flatten_nearest_k;
    flatten_nearest_k.reserve(k);
    flatten_nearest_k.insert(
      flatten_nearest_k.end(), local_nearest_k.begin(), local_nearest_k.end());

    size_t gather_size = workers * k;
    vector<feature_id_t> gathered_nearest_k;
    if (rank == 0)
      gathered_nearest_k.resize(gather_size);
    MPI_Gather(
      flatten_nearest_k.data(), k, MPI_UINT64_T, gathered_nearest_k.data(),
      gather_size, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    return feature_id_set_t(
      gathered_nearest_k.begin(), gathered_nearest_k.end());
  }

  feature_id_set_t mpi_brute_force_nearest_k(
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

    auto subworker_train_features =
      scatter_train_feature_ids(rank, workers, train_feature_ids);
    load_cache(subworker_train_features);
    auto local_nearest_k = local_brute_force_nearest_k(k, query_set);
    auto gathered_nearest_k =
      gather_nearest_k(rank, workers, k, local_nearest_k);

    auto s = gathered_nearest_k.begin();
    auto e = s;
    std::advance(e, k);
    return feature_id_set_t(s, e);
  }
}  // namespace dknn
