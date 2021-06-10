#include "dknn/dknn.hpp"
#include <mpi.h>

#include <boost/format.hpp>

namespace dknn {
  using std::string;
  using std::vector;

  struct mpi_knn_query_result_entry_t {
    feature_id_t feature_id;
    feature_distance_t distance;
    feature_class_t feature_class;
  };

  static MPI_Datatype __mpi_knn_query_result_entry_t__ = MPI_DATATYPE_NULL;

  bool scatter_gather_init() {
    MPI_Aint displacements[3];
    MPI_Aint base_address;

    mpi_knn_query_result_entry_t dummy;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&dummy.feature_id, &displacements[0]);
    MPI_Get_address(&dummy.distance, &displacements[1]);
    MPI_Get_address(&dummy.feature_class, &displacements[2]);

    for (auto& displacement : displacements)
      displacement = MPI_Aint_diff(displacement, base_address);

    int lengths[3] = {1, 1, 1};

    MPI_Datatype fields[3] = {MPI_UINT64_T, MPI_DOUBLE, MPI_UINT64_T};
    MPI_Type_create_struct(
      3, lengths, displacements, fields, &__mpi_knn_query_result_entry_t__);
    return MPI_Type_commit(&__mpi_knn_query_result_entry_t__) == MPI_SUCCESS;
  }

  vector<feature_id_t> scatter(feature_id_set_t const& train_feature_ids) {
    vector<feature_id_t> flatten_train_feature_ids;
    if (node_rank() == 0) {
      flatten_train_feature_ids.reserve(train_feature_ids.size());
      flatten_train_feature_ids.insert(
        flatten_train_feature_ids.end(), train_feature_ids.begin(),
        train_feature_ids.end());
    }

    vector<feature_id_t> subworker_train_feature_ids;
    int subworker_trainset_size = train_feature_ids.size() / world_size();
    if (subworker_trainset_size * world_size() != train_feature_ids.size()) {
      std::cerr << "train feature size should be multiple of world size"
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

  static vector<mpi_knn_query_result_entry_t> flatten_scattered(
    knn_set_query_result_t const& scattered_knn_results,
    size_t reserve_size = 0) {
    vector<mpi_knn_query_result_entry_t> result;
    result.reserve(reserve_size);
    for (auto const& knn_result : scattered_knn_results) {
      for (auto const& data : knn_result) {
        auto const& [id, distance, _class] = data;
        result.emplace_back(
          mpi_knn_query_result_entry_t {id, distance, _class});
      }
    }
    return result;
  }

  static knn_set_query_result_t unflatten_gathered(
    size_t k, size_t query_set_size,
    vector<mpi_knn_query_result_entry_t> const& gathered_flat_knn_results) {
    size_t scattered_data_size = k * query_set_size;

    knn_set_query_result_t gathered_knn_results;
    gathered_knn_results.resize(query_set_size);
    for (int r = 0; r < world_size(); r++) {
      for (size_t i = 0; i < query_set_size; i++) {
        auto& knn_result = gathered_knn_results[i];

        auto s = &gathered_flat_knn_results[scattered_data_size * r + k * i];
        auto e = s + k;
        for (auto p = s; p < e; p++) {
          auto const& data = *p;
          knn_result.emplace_back(feature_match_info_t {
            data.feature_id, data.distance, data.feature_class});
        }
      }
    }
    return gathered_knn_results;
  }

  knn_set_query_result_t gather(
    size_t k, size_t query_set_size,
    knn_set_query_result_t const& scattered_results) {
    if (query_set_size != scattered_results.size()) {
      std::cerr << "query set size and local knn result size mismatches!"
                << std::endl;
      return {};
    }
    size_t scattered_data_size = k * query_set_size;
    size_t gathered_data_size = world_size() * scattered_data_size;

    auto scattered_flat_knn_results =
      flatten_scattered(scattered_results, scattered_data_size);

    vector<mpi_knn_query_result_entry_t> gathered_flat_knn_results;
    if (node_rank() == 0)
      gathered_flat_knn_results.resize(gathered_data_size);

    auto scattered_data = scattered_flat_knn_results.data();
    auto gathered_data = gathered_flat_knn_results.data();
    MPI_Gather(
      scattered_data, scattered_data_size, __mpi_knn_query_result_entry_t__,
      gathered_data, scattered_data_size, __mpi_knn_query_result_entry_t__, 0,
      MPI_COMM_WORLD);
    if (node_rank() != 0)
      return {};  // empty for non-master nodes.

    return unflatten_gathered(k, query_set_size, gathered_flat_knn_results);
  }
}  // namespace dknn
