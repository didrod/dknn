#include "dknn/dknn.hpp"
#include <iostream>

int main(int argc, char** argv) {
  if (!dknn::init(&argc, &argv)) {
    std::cerr << "DKNN cluster init failed" << std::endl;
    return -1;
  }

  auto query_results = dknn::mpi_brute_force_nearest_k(
    10,
    {
      {0., 0., 0.},
    });

  dknn::term();
  return 0;
}
