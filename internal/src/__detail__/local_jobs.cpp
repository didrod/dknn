#include "dknn/__detail__/local_jobs.hpp"
#include "dknn/__detail__/dataset.hpp"

#include <boost/container/flat_map.hpp>
#include <omp.h>

#include <algorithm>
#include <tuple>
#include <iostream>

namespace dknn {
  static inline std::vector<feature_match_info_t> calculate_feature_matches(
    feature_t const& query_feature) {
    size_t const N = __feature_dataset__.size();

    std::vector<feature_match_info_t> matches;
    matches.resize(N);

#pragma omp parallel for num_threads(4)
    for (size_t i = 0; i < N; i++) {
      auto const& [id, data] = *__feature_dataset__.nth(i);
      auto const& [_class, train_feature] = data;

      double distance = 0.;
      for (size_t i = 0; i < train_feature.size(); i++) {
        auto const& c_q = query_feature.at(i);
        auto const& c_t = train_feature.at(i);
        auto const delta = c_t - c_q;
        distance += delta * delta;
      }
      feature_class_t feature_class = _class;
      matches[i] = {id, distance, feature_class};
    }
    return matches;
  }

  knn_query_result_t nearest_k(size_t k, feature_t const& query_feature) {
    if (k > __feature_dataset__.size()) {
      auto msg = "K is greater than train set size";
      std::cerr << msg << std::endl;
      std::cerr << "k: " << k << std::endl;
      std::cerr << "train set size: " << __feature_dataset__.size()
                << std::endl;
      // TODO
      return {};
    }

    auto matches = calculate_feature_matches(query_feature);
    if (matches.size() != __feature_dataset__.size()) {
      auto msg = "incorrect number of matches during scattered KNN search.";
      std::cerr << msg << std::endl;
      std::cerr << "matches.size(): " << matches.size() << std::endl;
      std::cerr << "train set size: " << __feature_dataset__.size()
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

    return knn_query_result_t(matches.begin(), matches.begin() + k);
  }

  knn_set_query_result_t nearest_k(size_t k, feature_set_t const& query_set) {
    knn_set_query_result_t result;
    for (auto const& query_feature : query_set)
      result.emplace_back(nearest_k(k, query_feature));
    return result;
  }
}  // namespace dknn
