#include "dknn/local_knn.hpp"

#include <boost/container/flat_map.hpp>
#include <omp.h>

#include <algorithm>
#include <tuple>
#include <iostream>

namespace dknn {
  id_feature_dict_t __local_train_feature_cache__ = {};

  void load_train_data(std::vector<feature_id_t> const& ids_to_load) {
    // TODO
  }

  using id_tagged_feature_distance_t = std::tuple<feature_id_t, double>;

  static inline std::vector<id_tagged_feature_distance_t>
  calculate_feature_distances(feature_t const& query_feature) {
    std::vector<id_tagged_feature_distance_t> distances;
    // TODO
    return distances;
  }

  template <typename feature_container_it_t>
  static inline feature_id_set_t feature_id_set_of(
    feature_container_it_t s, feature_container_it_t e) {
    feature_id_set_t result;
    for (auto i = s; i != e; i++) {
      auto const& [id, _] = *i;
      result.emplace(id);
    }
    return result;
  }

  template <typename feature_container_t>
  static inline feature_id_set_t feature_id_set_of(
    feature_container_t const& container) {
    return feature_id_set_of(container.begin(), container.end());
  }

  feature_id_set_t node_local_nearest_k(
    size_t k, feature_t const& query_feature) {
    if (k <= __local_train_feature_cache__.size())
      return feature_id_set_of(__local_train_feature_cache__);

    auto distances = calculate_feature_distances(query_feature);
    if (distances.size() != __local_train_feature_cache__.size()) {
      auto msg = "incorrect distance calculation during local KNN search.";
      std::cerr << msg << std::endl;
      std::cerr << "distances.size(): " << distances.size() << std::endl;
      std::cerr << "train set size: " << __local_train_feature_cache__.size()
                << std::endl;
      return {};
    }

    auto comparison_function = [](auto const& a, auto const& b) {
      auto const& [id_a, distance_a] = a;
      auto const& [id_b, distance_b] = b;
      return distance_a <= distance_b;
    };
    std::partial_sort(
      distances.begin(), distances.begin() + k, distances.end(),
      comparison_function);
    return feature_id_set_of(distances.begin(), distances.begin() + k);
  }

  std::vector<feature_id_set_t> node_local_nearest_k(
    size_t k, feature_set_t const& query_set) {
    std::vector<feature_id_set_t> result;
    for (auto const& query_feature : query_set)
      result.emplace_back(node_local_nearest_k(k, query_feature));
    return result;
  }
}  // namespace dknn
