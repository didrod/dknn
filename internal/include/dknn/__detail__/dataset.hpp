#pragma once

#include "dknn/typedefs.hpp"
#include <boost/container/flat_map.hpp>
#include <tuple>

namespace dknn {
  using class_annotated_feature_data_t = std::tuple<feature_class_t, feature_t>;

  using boost::container::flat_map;
  using feature_dataset_t =
    flat_map<feature_id_t, class_annotated_feature_data_t>;
  extern feature_dataset_t __feature_dataset__;

  void load_feature_dataset(std::string filename);
}  // namespace dknn
