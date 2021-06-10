#pragma once

#include "dknn/typedefs.hpp"
#include <boost/container/flat_map.hpp>

namespace dknn {
  using boost::container::flat_map;
  using id_feature_dict_t = flat_map<feature_id_t, feature_t>;
  extern id_feature_dict_t __local_train_feature_cache__;

  void load_feature_dataset();
}  // namespace dknn
