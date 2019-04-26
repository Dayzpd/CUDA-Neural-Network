#include "Dataset.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <math.h>
#include <memory>
#include <stdexcept>
#include <stdlib.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

namespace neural_network
{
  /// <summary><c>Dataset</c> constructor facilitates allocation of class
  /// properties.</summary>
  Dataset::Dataset(DatasetConfig& config) : config(config)
  {
    classes = std::vector<std::string>(config.num_classes);

    train_batches = std::vector<Neurons>(config.num_batches);
    train_targets = std::vector<Neurons>(config.num_batches);

    test_features = std::vector<Neurons>(config.num_test_features);
    test_targets = std::vector<Neurons>(config.num_test_features);

    allocate_test_features();

    allocate_train_features();

    register_classes(config.dataset_path + "/object_labels.csv");

    load_batches(config.dataset_path + "/batches/");

  }

  /// <summary><c>Dataset</c> destructor deletes all class properties.
  /// </summary>
  Dataset::~Dataset()
  {
    delete config;

    delete classes;

    for (auto feature : test_features)
    {
      delete feature;
    }

    for (auto target : test_targets)
    {
      delete target;
    }

    for (auto batch : train_batches)
    {
      delete batch;
    }

    for (auto target : train_targets)
    {
      delete target;
    }
  }

  /// <summary><c>allocate_test_features</c> fills both <c>test_features</c>
  /// and <c>test_targets</c> with Neurons objects of size (1, feature_size) and
  /// (1, num_classes).</summary>
  void Dataset::allocate_test_features()
  {
    for (size_t i = 0; i < num_test_features; i++)
    {
      test_features.push_back(new Neurons(1, config.feature_size));
      test_targets.push_back(new Neurons(1, config.num_classes));
    }
  }

  /// <summary><c>allocate_train_features</c> fills both <c>train_features</c>
  /// and <c>train_targets</c> with Neurons objects of size
  /// (batch_size, feature_size) and (batch_size, num_classes).
  void Dataset::allocate_train_features()
  {
    for (size_t i = 0; i < config.num_batches; i++)
    {
      train_features.push_back(
        new Neurons(config.batch_size, config.feature_size));
      train_targets.push_back(
        new Neurons(config.batch_size, config.num_classes));
    }
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
    std::ifstream test_features_file(labels_path);

    if (!test_features_file.is_open())
    {
      throw runtime_error("Error (Dataset::load_test): Failed to open test"
        + " features file.\n");
    }

    std::string value_str;
    int value_int;

    int feat_num = 0;

    // Read file until end or reached last feature.
    while (
      test_features_file.good() || feat_num < this->config.num_test_features
    ) {
      // Info for a single single spans two lines. For loop reads line one for
      // pixel values.
      for (int pixel = 0; pixel < this->config.feature_size; pixel++)
      {
        if (pixel < this->config.feature_size - 1)
        {
          getLine(test_features_file, value_str, ",");
        }
        else
        {
          getLine(test_features_file, value_str, "\n");
        }

        // Convert the string read in and convert to int.
        value_int = std::stoi(value_str, nullptr, 10);

        // Normalize the pixel value and place in the host_data for the current
        // feature.
        this->test_features.at(feat_num).host_data.at(pixel) = value_int / 255.0;
      }

      // Retrieve the class index and convert to integer.
      getLine(test_features_file, value_str, "\n");
      value_int = std::stoi(value_str, nullptr, 10);

      // Encode class index as one hot and copy to host_data target for the
      // current feature.
      std::vector<int> one_hot = encode_one_hot(value_int);

      thrust::copy(one_hot.begin(), one_hot.end(),
        test_targets.at(feat_num).host_data.begin());

      feat_num += 1;
    }

    objects_file.close();
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
      std::ifstream batch_file(
        batches_path + "batch_" + std::to_string(batch_num) + ".csv");

      if (!batch_file.is_open())
      {
        throw runtime_error("Error (Dataset::load_batches): Failed to open"
          + " batch file (Batch number: " + std::to_string(batch_num) + ").\n");
      }

      feat_num = 0;
      thrust::host_vector<float>::iterator targets_iterator
        = train_targets.at(batch_num).host_data.begin();

      // Read file until end or reached last feature.
      while (
        batch_file.good() || feat_num < config.batch_size
      ) {
        // Info for a single single spans two lines. For loop reads line one for
        // pixel values.
        for (int pixel = 0; pixel < this->config.feature_size; pixel++)
        {
          if (pixel < this->config.feature_size - 1)
          {
            getLine(batch_file, value_str, ",");
          }
          else
          {
            getLine(batch_file, value_str, "\n");
          }

          // Convert the string read in and convert to int.
          value_int = std::stoi(value_str, nullptr, 10);

          // Calulate host_data index
          int host_data_index = feat_num * config.feature_size + pixel;

          // Normalize the pixel value and place in the host_data for the
          // current feature.
          this->train_batches.at(batch_num).host_data.at(host_data_index)
            = value_int / 255.0;
        }

        // Retrieve the class index and convert to integer.
        getLine(test_features_file, value_str, "\n");
        value_int = std::stoi(value_str, nullptr, 10);

        // Encode class index as one hot and copy to host_data target for the
        // current feature.
        std::vector<int> one_hot = encode_one_hot(value_int);

        thrust::copy(one_hot.begin(), one_hot.end(),
          targets_iterator);

        targets_iterator += one_hot.size();

        feat_num += 1;
      }

      batch_file.close();
    }
  }

  /// <summary><c>register_classes</c> will load the object labels CSV file
  /// and assign class names to their corresponding indexes in the
  /// <c>classes</c> vector.</summary>
  void Dataset::register_classes(std::string objects_path)
  {
    std::ifstream objects_file(objects_path);

    if (!objects_file.is_open())
    {
      throw runtime_error("Error (Dataset::register_classes): Failed to load"
        + " object labels file.\n");
    }

    std::string class_name;
    std::string class_index_str;

    while (objects_file.good())
    {
      getLine(objects_file, class_name, ",");
      getLine(objects_file, class_index, "\n");

      int class_index_int = std::stoi(class_index_str, nullptr, 10);

      if (class_index_int >= classes.size())
      {
        throw runtime_error("Error (Dataset::register_classes): Object labels"
          + " file contains an index value greater than or equal to the number"
          + " of user-specified classes. Ensure that class indexing begins at"
          + " 0 and all following classes increment their index by 1.\n");
      }

      classes.at(class_index_int) = class_name;
    }

    objects_file.close();
  }

  /// <summary><c>decode_one_hot</c> finds and returns the index that has a
  /// value of 1.</summary>
  int Dataset::decode_one_hot(std::vector one_hot)
  {
    if (one_hot.size() != classes.size())
    {
      throw runtime_error("Error (Dataset::decode_one_hot): one hot vector"
        + " supplied to the method has size not equal to the number of classes"
        + " registered in the dataset (one_hot_vector size: "
        + std::to_string(one_hot.size()) + ", classes size: "
        + std::to_string(classes.size()) + ").\n");
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
      throw runtime_error("Error (Dataset::decode_one_hot): one hot vector"
        + " supplied to the method contains no value equal to 1.\n");
    }

    return class_index;
  }

  /// <summary><c>encode_one_hot</c> constructs a vector with a 1 at the index
  /// provided to the method and 0s in every other index.</summary>
  std::vector<int> Dataset::encode_one_hot(int class_index)
  {
    if (class_index < 0 || class_index >= classes.size())
    {
      throw runtime_error("Error (Dataset::encode_one_hot): class_index"
        + " supplied to the method is not within the range: [0, "
        + std::to_string(classes.size() - 1) + "]. Class index provided: "
        + std::to_string(class_index) + ".\n");
    }

    std::vector<int> one_hot_vector(classes.size(), 0);
    one_hot_vector.at(class_index) = 1;
    return one_hot_vector;
  }

  /// <summary><c>get_batch</c> returns a single requested batch.</summary>
  Neurons& get_batch(int batch_num)
  {
    if (batch_num >= config.num_batches || batch_num < 0)
    {
      throw runtime_error("Error (Dataset::get_batch): A batch was requested"
        + " that does not exist.\n"
        + "Valid input range: [ 0, " + std::to_string(config.num_batches - 1)
        + " ]\n"
        + "Batch number requested: " + std::to_string(batch_num) + "\n");
    }

    return train_batches.at(batch_num);
  }

  /// <summary><c>get_batch_targets</c> returns the one hot encoded targets
  /// for the requested batch.</summary>
  Neurons& get_batch_targets(int batch_num)
  {
    if (batch_num >= config.num_batches || batch_num < 0)
    {
      throw runtime_error("Error (Dataset::get_batch_targets): Batch targets
        + " were requested that do not exist.\n"
        + "Valid input range: [ 0, " + std::to_string(config.num_batches - 1)
        + " ]\n"
        + "Batch number requested: " + std::to_string(batch_num) + "\n");
    }

    return train_targets.at(batch_num);
  }

  /// <summary><c>get_test_feature</c> returns a single test feature.
  /// </summary>
  Neurons& get_test_feature(int feature_num)
  {
    if (feature_num >= config.num_test_features || feature_num < 0)
    {
      throw runtime_error("Error (Dataset::get_test_feature): A test feature"
        + " was requested that does not exist.\n"
        + "Valid input range: [ 0, "
        + std::to_string(config.num_test_features - 1) + " ]\n"
        + "Feature number requested: " + std::to_string(feature_num) + "\n");
    }

    return test_features.at(batch_num);
  }

  /// <summary><c>get_test_feature</c> returns the one hot encoded target
  /// for a single test feature.</summary>
  Neurons& get_test_target(int feature_num)
  {
    if (feature_num >= config.num_test_features || feature_num < 0)
    {
      throw runtime_error("Error (Dataset::get_test_feature): A test feature"
        + " target was requested that does not exist.\n"
        + "Valid input range: [ 0, "
        + std::to_string(config.num_test_features - 1) + " ]\n"
        + "Feature number requested: " + std::to_string(feature_num) + "\n");
    }

    return test_targets.at(batch_num);
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

    throw runtime_error("Error (Dataset::get_class_index): class_name"
      + " supplied to the method has not been registered. Class name provided: "
      + class_name + ".\n");
  }

  /// <summary><c>get_class_index</c> returns the class name of a class index.
  /// If the class index is not within the index range of classes, a
  /// runtime_error is thrown.</summary>
  int Dataset::get_class_name(int class_index)
  {
    if (class_index < 0 || class_index >= classes.size())
    {
      throw runtime_error("Error (Dataset::get_class_name): class_index"
        + " supplied to the method is not within the range: [0, "
        + std::to_string(classes.size() - 1) + "]. Class index provided: "
        + std::to_string(class_index) + ".\n");
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

}
