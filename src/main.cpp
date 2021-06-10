#include "dknn/dknn.hpp"
#include <iostream>

int main(int argc, char** argv) {
  if (!dknn::init(&argc, &argv)) {
    std::cerr << "DKNN cluster init failed" << std::endl;
    return -1;
  }

  dknn::term();
  return 0;
}
