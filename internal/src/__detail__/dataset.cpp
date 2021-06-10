#include "dknn/__detail__/dataset.hpp"
#include <sstream>
#include <fstream>
#include <iostream>

namespace dknn {
  feature_dataset_t __feature_dataset__ = {};

  // csv parser : from http://www.zedwood.com/article/cpp-csv-parser
  static std::vector<std::string> csv_read_row(
    std::istream& in, char delimiter) {
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

  void load_feature_dataset(std::string filename) {
    __feature_dataset__.clear();  // clear currently remaning data

    if (filename.empty())
      return;
    std::ifstream file(filename);

    int lineno = 0;
    while (file.good()) {
      auto row = csv_read_row(file, ',');
      int c = row.size();

      if (c < 2) {
        std::cerr << "reading CSV, but encountered invalid data ";
        std::cerr << "at line " << lineno << "; ";
        std::cerr << "each row should contain at least two fields ";
        std::cerr << "(ID and classID), but given row has only ";
        std::cerr << row.size() << " fields." << std::endl;
        std::cerr << "skipping..." << std::endl;
        continue;
      }

      feature_id_t id = std::stoi(row.at(c - 1));
      feature_class_t _class = std::stoi(row.at(c - 2));

      feature_t feature;
      for (size_t i = 0; i < row.size() - 2; i++)
        feature.emplace_back(std::stod(row.at(i)));

      __feature_dataset__.emplace(id, std::make_tuple(_class, feature));
    }
  }
}  // namespace dknn
