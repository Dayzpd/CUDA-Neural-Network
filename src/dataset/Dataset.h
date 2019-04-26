
#ifndef DATASET_H
#define DATASET_H

#include "DatasetConfig.h"
#include "../neurons/Neurons.h"

#include <map>
#include <string>
#include <vector>

namespace neural_network
{

  class Dataset
  {
    private:
      DatasetConfig& config;

      std::vector<std::string> classes; // uses index to access class name

      std::vector<Neurons> test_features; // features to test network
      std::vector<Neurons> test_targets; // feature targets for testing

      std::vector<Neurons> train_batches; // batches of features for training
      std::vector<Neurons> train_targets; // batch targets for training

      /// <summary><c>allocate_test_features</c> allocates memory for features
      /// used to test a neural network after it has been trained.</summary>
      void allocate_test_features();

      /// <summary><c>allocate_train_features</c> allocates memory for features
      /// used to train a neural network.</summary>
      void allocate_train_features();

      /// <summary><c>load_batches</c> loads train features into the
      /// <c>train_batches</c> vector along with each feature's target one hot
      /// encoded into the <c>train_targets</c> vector.</summary>
      /// <param name="batches_path">Absolute path to batch CSV file directory.
      /// </param>
      void load_batches(std::string batches_path);

      /// <summary><c>load_test</c> loads test features into the
      /// <c>test_features</c> vector along with each feature's target one hot
      /// encoded into the <c>test_targets</c> vector.</summary>
      /// <param name="test_set_path">Absolute path to test features CSV file.
      /// </param>
      void load_test(std::string test_set_path);

      /// <summary><c>register_classes</c> loads object labels into the
      /// <c>classes</c> vector.</summary>
      /// <param name="objects_path">Absolute path to object labels CSV
      /// file.</param>
      void register_classes(std::string objects_path);

    public:
      Dataset(DatasetConfig& config);

      ~Dataset();

      /// <summary><c>decode_one_hot</c> receives a one hot vector and returns
      /// its equivalent class index.</summary>
      /// <example>
      /// one_hot_vector: [ 0, 0, 1, 0, 0 ]
      /// class_index: 2
      /// </example>
      /// <param name="one_hot">One hot vector representing a
      /// pre-registered class object.</param>
      /// <returns>Class index represented by the one hot vector.</returns>
      int decode_one_hot(std::vector one_hot);

      /// <summary><c>encode_one_hot</c> receives a class index and returns its
      /// equivalent as a one hot vector.</summary>
      /// <example>
      /// num_classes: 5, class_index: 2
      /// output: [ 0, 0, 1, 0, 0 ]
      /// </example>
      /// <param name="class_index">Index value of a pre-registered class.
      /// </param>
      /// <returns>One hot encoded vector of supplied class index.</returns>
      std::vector<int> encode_one_hot(int class_index);

      /// <summary><c>get_batch</c> returns a single requested batch.</summary>
      /// <param name="batch_num">Batch number requested.</param>
      /// <returns>Neurons object for the requested batch.</returns>
      Neurons& get_batch(int batch_num);

      /// <summary><c>get_batch_targets</c> returns the one hot encoded targets
      /// for the requested batch.</summary>
      /// <param name="batch_num">Batch number requested.</param>
      /// <returns>Neurons object for the requested batch targets.</returns>
      Neurons& get_batch_targets(int batch_num);

      /// <summary><c>get_test_feature</c> returns a single test feature.
      /// </summary>
      /// <param name="feature_num">Test feature number requested.</param>
      /// <returns>Neurons object for the requested test feature.</returns>
      Neurons& get_test_feature(int feature_num);

      /// <summary><c>get_test_feature</c> returns the one hot encoded target
      /// for a single test feature.</summary>
      /// <param name="feature_num">Test feature number requested.</param>
      /// <returns>Neurons object for the requested test feature target.
      /// </returns>
      Neurons& get_test_target(int feature_num);

      /// <summary><c>get_class_index</c> receives a class name and returns its
      /// index.</summary>
      /// <param name="class_name">Class name for a given pre-registered object.
      /// </param>
      /// <returns>Index number of class name object.</returns>
      int get_class_index(std::string class_name);

      /// <summary><c>get_class_name</c> receives a class index and returns its
      /// name.</summary>
      /// <param name="class_index">Index value for a pre-registered object.
      /// </param>
      /// <returns>Class name of the supplied index.</returns>
      int get_class_name(int class_index);

      /// <summary><c>get_num_batches</c> returns number of batches.</summary>
      /// <returns>Number of batches.</returns>
      int get_num_batches();

      /// <summary><c>get_num_classes</c> returns number of class object types.
      /// </summary>
      /// <returns>Number of classes.</returns>
      int get_num_classes();

      /// <summary><c>get_num_test_features</c> returns number of features
      /// in the test set.</summary>
      /// <returns>Number of test features.</returns>
      int get_num_test_features();
  }

}
