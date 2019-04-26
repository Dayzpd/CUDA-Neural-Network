
#include <fstream>
#include <iostream>

void DatasetConfig::DatasetConfig(std::string dataset_path) :
  dataset_path(dataset_path)
{
  std::ifstream init_file(dataset_path + "/dataset_init.csv");

  if (!init_file.is_open())
  {
    throw runtime_error("Error (DatasetConfig::DatasetConfig): Failed to open "
      + "init file.\n");
  }

  std::string property;
  std::string value_str;

  while (init_file.good())
  {
    getLine(init_file, property, ",");
    getLine(init_file, value_str, "\n");

    int value_int = std::stoi(value_str, nullptr, 10);

    switch (property)
    {
      case "batch_size":
        this->batch_size = value_int;
        cout << "Batch Size: " << this->batch_size << endl;
        break;
      case "dim_size":
        this->dim_size = value_int;
        cout << "Dimension Size: " << this->dim_size << endl;
        this->feature_size = value_int * value_int;
        cout << "Feature Size: " << this->feature_size << endl;
        break;
      case "num_batches":
        this->num_batches = value_int;
        cout << "Number of Batches: " << this->num_batches << endl;
        break;
      case "num_classes":
        this->num_classes = value_int;
        cout << "Number of Classes: " << this->num_classes << endl;
        break;
      case "num_test_features":
        this->num_test_features = value_int;
        cout << "Number of Test Features: " << this->num_test_features << endl;
        break;
    }
  }

  init_file.close();
}
