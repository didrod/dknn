#include "dknn/__detail__/local_jobs.hpp"

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

  using id_tagged_feature_match_t =
    std::tuple<feature_id_t, feature_distance_t, feature_class_t>;

  static inline std::vector<id_tagged_feature_match_t>
  calculate_feature_matches(feature_t const& query_feature) {
    size_t const N = __local_train_feature_cache__.size();

    std::vector<id_tagged_feature_match_t> matches;
    matches.resize(N);

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
      feature_class_t feature_class = 0;  // TODO
      matches[i] = {id, distance, feature_class};
    }
    return matches;
  }

  knn_query_result_t nearest_k(size_t k, feature_t const& query_feature) {
    if (k > __local_train_feature_cache__.size()) {
      auto msg = "K is greater than train set size";
      std::cerr << msg << std::endl;
      std::cerr << "k: " << k << std::endl;
      std::cerr << "train set size: " << __local_train_feature_cache__.size()
                << std::endl;
      // TODO
      return {};
    }

    auto matches = calculate_feature_matches(query_feature);
    if (matches.size() != __local_train_feature_cache__.size()) {
      auto msg = "incorrect number of matches during scattered KNN search.";
      std::cerr << msg << std::endl;
      std::cerr << "matches.size(): " << matches.size() << std::endl;
      std::cerr << "train set size: " << __local_train_feature_cache__.size()
                << std::endl;
      return {};
    }

    auto comparison_function = [](auto const& a, auto const& b) {
      auto const& [id_a, distance_a, class_a] = a;
      auto const& [id_b, distance_b, class_b] = b;
      return distance_a <= distance_b;
    };
    std::partial_sort(
      matches.begin(), matches.begin() + k, matches.end(), comparison_function);

    knn_query_result_t result;
    for (size_t i = 0; i < k; i++) {
      auto const& [id, distance, cls] = matches.at(i);
      result.emplace(id, feature_match_info_t {distance, cls});
    }
    return result;
  }

  knn_set_query_result_t nearest_k(size_t k, feature_set_t const& query_set) {
    knn_set_query_result_t result;
    for (auto const& query_feature : query_set)
      result.emplace_back(nearest_k(k, query_feature));
    return result;
  }
}  // namespace dknn
