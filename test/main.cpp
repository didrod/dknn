#include "dknn/dknn.hpp"
#include <iostream>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

int main(int argc, char* argv[]) {
  if (!dknn::init(&argc, &argv)) {
    std::cerr << "DKNN cluster init failed" << std::endl;
    return -1;
  }
  int result = Catch::Session().run(argc, argv);
  dknn::term();
  return result;
}
