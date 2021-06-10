#pragma once

#include "dknn/typedefs.hpp"

namespace dknn {
  void send_query(feature_set_t const& query_set);
  feature_set_t receive_query();
}  // namespace dknn
