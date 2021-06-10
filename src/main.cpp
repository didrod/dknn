#include "dknn/dknn.hpp"
#include <iostream>
#include <sstream>
#include <fstream>

// csv parser : from http://www.zedwood.com/article/cpp-csv-parser
static std::vector<std::string> csv_read_row(std::istream& in, char delimiter) {
  std::stringstream ss;
  bool inquotes = false;
  std::vector<std::string> row;  // relying on RVO
  while (in.good()) {
    char c = in.get();
    if (!inquotes && c == '"') {  // beginquotechar
      inquotes = true;
    } else if (inquotes && c == '"') {  // quotechar
      if (in.peek() == '"') {  // 2 consecutive quotes resolve to 1
        ss << (char)in.get();
      } else {  // endquotechar
        inquotes = false;
      }
    } else if (!inquotes && c == delimiter) {  // end of field
      row.push_back(ss.str());
      ss.str("");
    } else if (!inquotes && (c == '\r' || c == '\n')) {
      if (in.peek() == '\n') {
        in.get();
      }
      row.push_back(ss.str());
      break;
    } else {
      ss << c;
    }
  }
  return row;
}

static auto load_class_truths(std::string path) {
  std::ifstream file(path);

  std::vector<dknn::feature_class_t> class_truths;
  while (file.good()) {
    auto row = csv_read_row(file, ',');
    int c = row.size();

    dknn::feature_class_t _class = std::stoi(row.at(c - 2));
    class_truths.emplace_back(_class);
  }
  return class_truths;
}

static auto load_query_set(std::string path) {
  std::ifstream file(path);

  dknn::feature_set_t query_features;
  while (file.good()) {
    auto row = csv_read_row(file, ',');
    int c = row.size();

    dknn::feature_t feature;
    for (size_t i = 0; i < row.size() - 2; i++)
      feature.emplace_back(std::stod(row.at(i)));

    query_features.emplace_back(feature);
  }
  return query_features;
}

int main(int argc, char** argv) {
  if (!dknn::init(&argc, &argv)) {
    std::cerr << "DKNN cluster init failed" << std::endl;
    return -1;
  }

  auto query_file = argv[2];
  auto query_results = dknn::mpi_brute_force_nearest_k(
    10, [query_file]() { return load_query_set(query_file); });
  auto truths = load_class_truths(query_file);

  if (dknn::node_rank() == 0) {
    std::cout << "classification results:" << std::endl;
    for (auto _class : query_results)
      std::cout << _class << ", ";
    std::cout << std::endl;

    std::cout << "truths:" << std::endl;
    for (auto _class : truths)
      std::cout << _class << ", ";
    std::cout << std::endl;
  }

  dknn::term();
  return 0;
}
