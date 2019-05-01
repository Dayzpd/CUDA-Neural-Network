
#include "DatasetConfig.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace cuda_net
{

  DatasetConfig::DatasetConfig(std::string dataset_path) :
    dataset_path(dataset_path)
  {
    std::ifstream file(dataset_path + "/dataset_init.csv");

    if (!file.is_open())
    {
      throw std::runtime_error(
        "Error (DatasetConfig::DatasetConfig): Failed to open init file.\n"
      );
    }

    std::string property;
    std::string value_str;

    while (file.good())
    {
      std::getline(file, property, ',');
      std::getline(file, value_str, '\n');

      int value_int = std::stoi(value_str);

      if (property == "batch_size")
      {
          this->batch_size = value_int;
          std::cout << "Batch Size: " << this->batch_size << std::endl;
          continue;
      }

      if (property == "dim_size")
      {
          this->dim_size = value_int;
          std::cout << "Dimension Size: " << this->dim_size << std::endl;
          this->feature_size = value_int * value_int;
          std::cout << "Feature Size: " << this->feature_size << std::endl;
          continue;
      }

      if (property == "num_batches")
      {
          this->num_batches = value_int;
          std::cout << "Number of Batches: " << this->num_batches << std::endl;
          continue;
      }

      if (property == "num_classes")
      {
          this->num_classes = value_int;
          std::cout << "Number of Classes: " << this->num_classes << std::endl;
          continue;
      }

      if (property == "num_test_features")
      {
          this->num_test_features = value_int;
          std::cout << "Number of Test Features: " << this->num_test_features
            << std::endl;
          continue;
      }

    }

    file.close();
  }

}
