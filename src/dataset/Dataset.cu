#include "Dataset.h"

#include <fstream>
#include <iostream>
#include <math.h>
#include <memory>
#include <stdexcept>
#include <stdlib.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

/// <summary><c>Dataset</c> constructor facilitates allocation of class
/// properties.</summary>
Dataset::Dataset(DatasetConfig& config) : config(config)
{
  classes = std::vector<std::string>(config.num_classes);

  allocate_test_features();

  allocate_train_features();

  register_classes(config.dataset_path + "/object_labels.csv");

  load_batches(config.dataset_path + "/batches/");

  load_test(config.dataset_path + "/test_features.csv");
}

/// <summary></summary>
Dataset::~Dataset()
{

}

/// <summary><c>allocate_test_features</c> fills both <c>test_features</c>
/// and <c>test_targets</c> with Neurons objects of size (1, feature_size) and
/// (1, num_classes).</summary>
void Dataset::allocate_test_features()
{
  std::cout << "Allocating space for test features..." << std::endl;

  for (size_t i = 0; i < config.num_test_features; i++)
  {
    test_features.push_back(Neurons(1, config.feature_size));
    test_features.at(i).allocate_memory();
    test_targets.push_back(Neurons(1, config.num_classes));
    test_targets.at(i).allocate_memory();
  }

  std::cout << "Finished allocating test features..." << std::endl;
}

/// <summary><c>allocate_train_features</c> fills both <c>train_features</c>
/// and <c>train_targets</c> with Neurons objects of size
/// (batch_size, feature_size) and (batch_size, num_classes).
void Dataset::allocate_train_features()
{
  std::cout << "Allocating space for train batches..." << std::endl;

  for (size_t i = 0; i < config.num_batches; i++)
  {
    train_batches.push_back(Neurons(config.batch_size, config.feature_size));
    train_batches.at(i).allocate_memory();
    train_targets.push_back(Neurons(config.batch_size, config.num_classes));
    train_targets.at(i).allocate_memory();
  }

  std::cout << "Finished allocating train batches..." << std::endl;
}

/// <summary><c>load_test</c> will load two rows for each iteration of the
/// while loop. The first row will get the pixel values of an image and the
/// second row contains the class index which will be converted into a one hot
/// vector. The pixel values will be placed inside the Neurons objects of
/// <c>test_features</c> and the one hot vector targets will be placed into
/// <c>test_targets</c>. Note that all pixel values are normalized by dividing
/// by 255.</summary>
void Dataset::load_test(std::string test_set_path)
{
  std::cout << "Loading test features csv file..." << std::endl;

  std::ifstream file(test_set_path);

  if (!file.is_open())
  {
    throw std::runtime_error(
      "Error (Dataset::load_test): Failed to open test features file.\n"
    );
  }

  std::string value_str;
  int value_int;

  int feat_num = 0;

  // Read file until end or reached last feature.
  while (feat_num < this->config.num_test_features)
  {
    // Info for a single single spans two lines. For loop reads line one for
    // pixel values.
    for (int pixel = 0; pixel < this->config.feature_size; pixel++)
    {
      if (pixel < this->config.feature_size - 1)
      {
        std::getline(file, value_str, ',');
      }
      else
      {
        std::getline(file, value_str, '\n');
      }

      if(value_str.find_first_not_of(' ') == std::string::npos)
      {
        break;
      }

      // Convert the string read in and convert to int.
      value_int = std::stoi(value_str);

      // Normalize the pixel value and place in the device_data for the current
      // feature.
      test_features.at(feat_num).device_data[pixel] = value_int / 255.0;
    }

    // Retrieve the class index and convert to integer.
    std::getline(file, value_str, '\n');

    while(value_str.find_first_not_of(' ') == std::string::npos)
    {
      std::getline(file, value_str, '\n');
      break;
    }

    value_int = std::stoi(value_str);

    // Encode class index as one hot and copy to device_data target for the
    // current feature.
    std::vector<float> one_hot = encode_one_hot(value_int);

    thrust::copy(one_hot.begin(), one_hot.end(),
    test_targets.at(feat_num).device_data.begin());

    feat_num += 1;
  }

  file.close();

  std::cout << "Finished loading test features csv file." << std::endl;
}

/// <summary><c>load_batches</c> will loop through each batch file. For each
/// file, the first row has the pixel values of an image and the second row
/// contains the class index which will be converted into a one hot
/// vector. The pixel values will be placed inside the Neurons objects of
/// <c>train_batches</c> and the one hot vector targets will be placed into
/// <c>train_targets</c>. Note that all pixel values are normalized by
/// dividing by 255.</summary>
void Dataset::load_batches(std::string batches_path)
{
  std::cout << "Loading train batches csv..." << std::endl;

  // Temporarily stores string value of pixel values and class indexes.
  std::string value_str;
  // Receives casted string to int value.
  int value_int;
  // Keeps track of feature number within each batch file.
  int feat_num;

  // Iterate through all batch files and read each files contents.
  for (int batch_num = 0; batch_num < config.num_batches; batch_num++)
  {
    // Build string for this iteration's batch file.
    std::ifstream file(
    batches_path + "batch_" + std::to_string(batch_num) + ".csv");

    if (!file.is_open())
    {
      throw std::runtime_error(
        "Error (Dataset::load_batches): Failed to open batch file. \
        \nBatch number: "
        + std::to_string(batch_num)
        + "\n"
      );
    }

    feat_num = 0;
    thrust::device_vector<float>::iterator targets_iterator
      = train_targets.at(batch_num).device_data.begin();

    // Read file until end or reached last feature.
    while (feat_num < config.batch_size)
    {
      // Info for a single single spans two lines. For loop reads line one for
      // pixel values.
      for (int pixel = 0; pixel < this->config.feature_size; pixel++)
      {
        if (pixel < this->config.feature_size - 1)
        {
          std::getline(file, value_str, ',');
        }
        else
        {
          std::getline(file, value_str, '\n');
        }

        if(value_str.find_first_not_of(' ') == std::string::npos)
        {
          break;
        }

        // Convert the string read in and convert to int.
        value_int = std::stoi(value_str);

        // Calulate device_data index
        int device_data_index = feat_num * config.feature_size + pixel;

        // Normalize the pixel value and place in the device_data for the
        // current feature.
        train_batches.at(batch_num).device_data[device_data_index]
          = value_int / 255.0;
      }

      // Retrieve the class index and convert to integer.
      std::getline(file, value_str, '\n');

      while(value_str.find_first_not_of(' ') == std::string::npos)
      {
        std::getline(file, value_str, '\n');
        break;
      }

      value_int = std::stoi(value_str);

      // Encode class index as one hot and copy to device_data target for the
      // current feature.
      std::vector<float> one_hot = encode_one_hot(value_int);

      thrust::copy(one_hot.begin(), one_hot.end(),
        targets_iterator);

      targets_iterator += one_hot.size();

      feat_num += 1;
    }

    file.close();
  }

  std::cout << "Finished loading train batches csv." << std::endl;
}

/// <summary><c>register_classes</c> will load the object labels CSV file
/// and assign class names to their corresponding indexes in the
/// <c>classes</c> vector.</summary>
void Dataset::register_classes(std::string objects_path)
{
  std::cout << "Loading class registration csv file..." << std::endl;

  std::ifstream file(objects_path);

  if (!file.is_open())
  {
    throw std::runtime_error(
      "Error (Dataset::register_classes): Failed to load object labels.\n"
    );
  }

  std::string class_name;
  std::string class_index_str;
  int class_index_int;

  while (file.good())
  {
    std::getline(file, class_name, ',');
    std::getline(file, class_index_str, '\n');

    int class_index_int = std::stoi(class_index_str);

    if (class_index_int >= classes.size())
    {
      throw std::runtime_error(
        "Error (Dataset::register_classes): Object labels file contains an \
        index value greater than or equal to the number of user-specified \
        classes. Ensure that class indexing begins at 0 and all following \
        classes increment their index by 1.\n"
      );
    }

    classes.at(class_index_int) = class_name;
  }

  file.close();

  std::cout << "Finished registration object classes." << std::endl;
}

/// <summary><c>decode_one_hot</c> finds and returns the index that has a
/// value of 1.</summary>
int Dataset::decode_one_hot(std::vector<float> one_hot)
{
  if (one_hot.size() != classes.size())
  {
    throw std::runtime_error(
      "Error (Dataset::decode_one_hot): one hot vector supplied to the \
      method has size not equal to the number of classes registered in the \
      dataset. \n one_hot_vector size: "
      + std::to_string(one_hot.size())
      + "\n classes size: "
      + std::to_string(classes.size())
      + "\n"
    );
  }

  int class_index = -1;

  for (size_t i = 0; i < one_hot.size(); i++)
  {
    if (one_hot.at(i) == 1)
    {
      class_index = i;
      break;
    }
  }

  if (class_index == -1)
  {
    throw std::runtime_error(
      "Error (Dataset::decode_one_hot): one hot vector supplied to the \
      method contains no value equal to 1.\n"
    );
  }

  return class_index;
}

/// <summary><c>encode_one_hot</c> constructs a vector with a 1 at the index
/// provided to the method and 0s in every other index.</summary>
std::vector<float> Dataset::encode_one_hot(int class_index)
{
if (class_index < 0 || class_index >= classes.size())
{
  throw std::runtime_error(
	"Error (Dataset::encode_one_hot): class_index supplied to the method \
	is not within the range: [0, "
	+ std::to_string(classes.size() - 1)
	+ "].\n Class index provided: "
	+ std::to_string(class_index)
	+ "\n"
  );
}

std::vector<float> one_hot_vector(classes.size(), 0);
one_hot_vector.at(class_index) = 1;
return one_hot_vector;
}

/// <summary><c>get_batch</c> returns a single requested batch.</summary>
Neurons& Dataset::get_batch(int batch_num)
{
  if (batch_num >= config.num_batches || batch_num < 0)
  {
    throw std::runtime_error(
      "Error (Dataset::get_batch): A batch was requested that does not \
      exist. Valid input range: [ 0, "
      + std::to_string(config.num_batches - 1)
      + " ].\n Batch number requested: "
      + std::to_string(batch_num)
      + "\n"
    );
  }

  return train_batches.at(batch_num);
}

/// <summary><c>get_batch_targets</c> returns the one hot encoded targets
/// for the requested batch.</summary>
Neurons& Dataset::get_batch_targets(int batch_num)
{
  if (batch_num >= config.num_batches || batch_num < 0)
  {
    throw std::runtime_error(
      "Error (Dataset::get_batch_targets): Batch targets were requested that \
      do not exist. Valid input range: [ 0, "
      + std::to_string(config.num_batches - 1)
      + " ].\n Batch number requested: "
      + std::to_string(batch_num)
      + "\n"
    );
  }

  return train_targets.at(batch_num);
}

/// <summary><c>get_test_feature</c> returns a single test feature.
/// </summary>
Neurons& Dataset::get_test_feature(int feature_num)
{
  if (feature_num >= config.num_test_features || feature_num < 0)
  {
    throw std::runtime_error(
      "Error (Dataset::get_test_feature): A test feature was requested that \
      does not exist.\n Valid input range: [ 0, "
      + std::to_string(config.num_test_features - 1)
      + " ].\n Feature number requested: "
      + std::to_string(feature_num)
      + "\n"
    );
  }

  return test_features.at(feature_num);
}

/// <summary><c>get_test_feature</c> returns the one hot encoded target
/// for a single test feature.</summary>
Neurons& Dataset::get_test_target(int feature_num)
{
  if (feature_num >= config.num_test_features || feature_num < 0)
  {
    throw std::runtime_error(
      "Error (Dataset::get_test_feature): A test feature target was \
      requested that does not exist. Valid input range: [ 0, "
      + std::to_string(config.num_test_features - 1)
      + " ].\n Feature number requested: "
      + std::to_string(feature_num)
      + "\n"
    );
  }

  return test_targets.at(feature_num);
}

/// <summary><c>get_class_index</c> loops through each string value in the
/// <c>classes</c> vector, and once a match is found, the index is returned.
/// If a match isn't found, a runtime_error is thrown.</summary>
int Dataset::get_class_index(std::string class_name)
{
  for (size_t i = 0; i < classes.size(); i++)
  {
    if (!class_name.compare(classes.at(i)))
    {
    return i;
    }
  }

  throw std::runtime_error(
    "Error (Dataset::get_class_index): class_name supplied to the method has \
    not been registered.\n Class name provided: "
    + class_name
    + ".\n"
  );
}

/// <summary><c>get_class_index</c> returns the class name of a class index.
/// If the class index is not within the index range of classes, a
/// runtime_error is thrown.</summary>
std::string Dataset::get_class_name(int class_index)
{
  if (class_index < 0 || class_index >= classes.size())
  {
    throw std::runtime_error(
      "Error (Dataset::get_class_name): class_index supplied to the method \
      is not within the range: [0, "
      + std::to_string(classes.size() - 1)
      + "].\n Class index provided: "
      + std::to_string(class_index)
      + ".\n"
    );
  }

  return classes.at(class_index);
}

/// <summary><c>get_num_batches</c> returns the size of the
/// <c>train_batches</c> vector.</summary>
int Dataset::get_num_batches()
{
  return train_batches.size();
}

/// <summary><c>get_num_classes</c> returns the size of the
/// <c>classes</c> vector.</summary>
int Dataset::get_num_classes()
{
  return classes.size();
}

/// <summary><c>get_num_test_features</c> returns the size of the
/// <c>test_features</c> vector.</summary>
int Dataset::get_num_test_features()
{
  return test_features.size();
}
