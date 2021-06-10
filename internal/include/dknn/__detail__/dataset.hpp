#pragma once

#include "dknn/typedefs.hpp"
#include <boost/container/flat_map.hpp>

namespace dknn {
  using boost::container::flat_map;
  using feature_dataset_t = flat_map<feature_id_t, feature_t>;
  extern feature_dataset_t __feature_dataset__;

  void load_feature_dataset();
}  // namespace dknn
