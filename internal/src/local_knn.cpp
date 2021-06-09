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
    size_t const N = __local_train_feature_cache__.size();

    std::vector<id_tagged_feature_distance_t> distances;
    distances.resize(N);

#pragma omp parallel for num_threads(4)
    for (size_t i = 0; i < N; i++) {
      auto const& [id, train_feature] = *__local_train_feature_cache__.nth(i);

      double distance = 0.;
      for (size_t i = 0; i < train_feature.size(); i++) {
        auto const& c_q = query_feature.at(i);
        auto const& c_t = train_feature.at(i);
        auto const delta = c_t - c_q;
        distance += delta * delta;
      }
      distances[i] = {id, distance};
    }
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
    if (k > __local_train_feature_cache__.size()) {
      auto msg = "K is greater than train set size";
      std::cerr << msg << std::endl;
      std::cerr << "k: " << k << std::endl;
      std::cerr << "train set size: " << __local_train_feature_cache__.size()
                << std::endl;
      return feature_id_set_of(__local_train_feature_cache__);
    }

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
