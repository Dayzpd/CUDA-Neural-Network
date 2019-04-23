#include "Dataset.h"

#include <assert.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <math.h>
#include <memory>
#include <stdexcept>
#include <stdlib.h>
#include <thrust/copy.h>

namespace neural_network
{
  /// <summary><c>Dataset</c> constructor facilitates allocation of class
  /// properties.</summary>
  Dataset::Dataset(int batch_size, int num_classes, int feature_size,
    int num_test_features, int num_train_features
  ) {
    classes = std::vector<std::string>(num_classes);

    test_features = std::vector<Neurons>(num_test_features);
    test_targets = std::vector<Neurons>(num_test_features);

    int num_batches = ceil(num_train_features / batch_size);

    train_batches = std::vector<Neurons>(num_batches);
    train_targets = std::vector<Neurons>(num_batches);

    allocate_test_features(num_classes, num_test_features, feature_size);

    allocate_train_features(num_classes, num_train_features, feature_size,
      num_batches, batch_size);
  }

  /// <summary><c>Dataset</c> deconstructor deletes all class properties.
  /// </summary>
  Dataset::~Dataset()
  {
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
  void allocate_test_features(int num_classes, int num_test_features,
    int feature_size
  ) {
    for (size_t i = 0; i < num_test_features; i++)
    {
      test_features.push_back(new Neurons(1, feature_size));
      test_targets.push_back(new Neurons(1, num_classes));
    }
  }

  /// <summary><c>allocate_train_features</c> fills both <c>test_features</c>
  /// and <c>test_targets</c> with Neurons objects of size
  /// (batch_size, feature_size) and (batch_size, num_classes). However, if
  /// <c>num_train_features / batch_size</c> produces a remainder,
  /// the X dimension of the last Neurons object in both afformentioned vectors
  /// will be equal to that remainder value.</summary>
  void allocate_train_features(int num_classes, int num_train_features,
    int feature_size, int num_batches, int batch_size
  ) {
    // Account for last batch size if remaining features exist.
    int last_batch_size = num_train_features % batch_size;

    if (last_batch_size == 0)
    {
      last_batch_size = batch_size
    }

    for (size_t i = 0; i < (num_batches - 1); i++)
    {
      train_features.push_back(new Neurons(batch_size, feature_size));
      train_targets.push_back(new Neurons(batch_size, num_classes));
    }

    train_features.push_back(new Neurons(last_batch_size, feature_size));
    train_targets.push_back(new Neurons(last_batch_size, num_classes));
  }

  /// <summary><c>decode_one_hot</c> finds and returns the index that has a
  /// value of 1.</summary>
  int decode_one_hot(std::vector one_hot_vector)
  {
    if (one_hot_vector.size() != classes.size())
    {
      throw runtime_error("Error (Dataset::decode_one_hot): one_hot_vector"
        + " supplied to the method has size not equal to the number of classes"
        + " registered in the dataset (one_hot_vector size: "
        + std::to_string(one_hot_vector.size()) + ", classes size: "
        + std::to_string(classes.size()) + ").\n");
    }

    int class_index = -1;

    for (size_t i = 0; i < one_hot_vector.size(); i++)
    {
      if (one_hot_vector.at(i) == 1)
      {
        class_index = i;
        break;
      }
    }

    if (class_index == -1)
    {
      throw runtime_error("Error (Dataset::decode_one_hot): one_hot_vector"
        + " supplied to the method contains no value equal to 1.\n");
    }

    return class_index;
  }

  /// <summary><c>encode_one_hot</c> constructs a vector with a 1 at the index
  /// provided to the method and 0s in every other index.</summary>
  std::vector<int> encode_one_hot(int class_index)
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

  /// <summary><c>get_class_index</c> loops through each string value in the
  /// <c>classes</c> vector, and once a match is found, the index is returned.
  /// If a match isn't found, a runtime_error is thrown.</summary>
  int get_class_index(std::string class_name)
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
  int get_class_name(int class_index)
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
  int get_num_batches()
  {
    return train_batches.size();
  }

  /// <summary><c>get_num_classes</c> returns the size of the
  /// <c>classes</c> vector.</summary>
  int get_num_classes()
  {
    return classes.size();
  }

  /// <summary><c>get_num_test_features</c> returns the size of the
  /// <c>test_features</c> vector.</summary>
  int get_num_test_features()
  {
    return test_features.size();
  }

  /// <summary><c>load_test</c> will first load the labels CSV file and then
  /// use the file name and object class to load images and one hot vectors into
  /// the Neurons objects in <c>test_features</c> and <c>test_targets</c>
  /// vectors.</summary>
  void Dataset::load_test(std::string labels_path, std::string dataset_path)
  {
    std::ifstream labels(labels_path);

    if (!labels.is_open())
    {
      throw runtime_error("Error (Dataset::load_test): Failed to open test"
        + " labels file.\n");
    }

    std::string file_name;
    std::string class_name;

    int line_num = 0;

    while (objects_file.good())
    {
      getLine(objects_file, file_name, ",");
      getLine(objects_file, class_name, "\n");

      int class_index = get_class_index(class_name);

      /*if (class_index < 0 || class_index >= classes.size())
      {
        throw runtime_error("Error (Dataset::load_test): class_index"
          + " from test label file line [" + std::to_string(line_num) + "] is"
          + " not within the range: [0, " + std::to_string(classes.size() - 1)
          + "]. Class index provided: " + std::to_string(class_index) + ".\n");
      }*/

      std::vector<int> one_hot = encode_one_hot(class_index);

      thrust::copy(one_hot.begin(), one_hot.end(),
        test_targets.at(line_num).begin());
    }

    objects_file.close();
  }

  /// <summary><c>load_train</c> will load each line of the labels CSV file,
  /// create a random permutation, and then load images and one hot vectors into
  /// the Neurons objects in <c>train_batches</c> and <c>train_targets</c>
  /// vectors.</summary>
  void Dataset::load_train(std::string labels_path, std::string dataset_path)
  {

  }

  /// <summary><c>register_classes</c> will load the object labels CSV file
  /// and assign class names to their corresponding indexes in the
  /// <c>classes</c> vector.</summary>
  void register_classes(std::string objects_path)
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
}
