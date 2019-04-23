
#ifndef MINI_BATCH_H
#define MINI_BATCH_H

#include "../neurons/Neurons.h"

#include <map>
#include <string>
#include <vector>

namespace neural_network
{

  class Dataset
  {
    private:
      std::vector<std::string> classes; // uses index to access class name

      std::vector<Neurons> test_features; // features to test network
      std::vector<Neurons> test_targets; // feature targets for testing

      std::vector<Neurons> train_batches; // batches of features for training
      std::vector<Neurons> train_targets; // batch targets for training

      /// <summary><c>allocate_test_features</c> allocates memory for features
      /// used to test a neural network after it has been trained.</summary>
      /// <param name="num_classes">Number of class object types.</param>
      /// <param name="num_test_features">Number of features used to test the
      /// neural network.</param>
      /// <param name="feature_size">Size of each feature (e.g. a 28 x 28 image
      /// has a feature size of 784).</param>
      void allocate_test_features(int num_classes, int num_test_features,
        int feature_size);

      /// <summary><c>allocate_train_features</c> allocates memory for features
      /// used to train a neural network.</summary>
      /// <param name="num_classes">Number of class object types.</param>
      /// <param name="num_train_features">Number of features used to train the
      /// neural network.</param>
      /// <param name="feature_size">Size of each feature (e.g. a 28 x 28 image
      /// has a feature size of 784).</param>
      /// <param name="num_batches">Number of training batches (e.g.
      /// <c>ceil(num_train_features / batch_size)</c>).</param>
      /// <param name="batch_size">Number of features in each batch.</param>
      void allocate_train_features(int num_classes, int num_train_features,
        int feature_size, int num_batches, int batch_size);

    public:
      Dataset(int batch_size, int num_classes, int feature_size,
        int num_test_features, int num_train_features);

      ~Dataset();

      /// <summary><c>decode_one_hot</c> receives a one hot vector and returns
      /// its equivalent class index.</summary>
      /// <example>
      /// one_hot_vector: [ 0, 0, 1, 0, 0 ]
      /// class_index: 2
      /// </example>
      /// <param name="one_hot_vector">One hot vector representing a
      /// pre-registered class object.</param>
      /// <returns>Class index represented by the one hot vector.</returns>
      int decode_one_hot(std::vector one_hot_vector);

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

      /// <summary><c>load_test</c> loads test features into the
      /// <c>test_features</c> vector along with each feature's target one hot
      /// encoded into the <c>test_targets</c> vector.</summary>
      /// <param name="labels_path">Absolute path to test feature labels CSV
      /// file. CSV file should be formmated as such:
      /// <example>
      /// 0.png,0
      /// 1.png,0
      /// 2.png,1
      /// 3.png,2
      /// ...
      /// </example>
      /// </param>
      /// <param name="dataset_path">Absolute path to directory containing test
      /// features.</param>
      void load_test(std::string labels_path, std::string dataset_path);

      /// <summary><c>load_train</c> loads train features into the
      /// <c>train_batches</c> vector along with each feature's target one hot
      /// encoded into the <c>train_targets</c> vector.</summary>
      /// <param name="labels_path">Absolute path to train feature labels CSV
      /// file. CSV file should be formmated as such:
      /// <example>
      /// 0.png,0
      /// 1.png,0
      /// 2.png,1
      /// 3.png,2
      /// ...
      /// </example>
      /// </param>
      /// <param name="dataset_path">Absolute path to directory containing train
      /// features.</param>
      void load_train(std::string labels_path, std::string dataset_path);

      /// <summary><c>register_classes</c> loads object labels into the
      /// <c>classes</c> vector.</summary>
      /// <param name="objects_path">Absolute path to object labels CSV
      /// file. CSV file should contain lines formatted as such:
      /// <example>
      /// bird,0
      /// deer,1
      /// person,2
      /// ...
      /// </example>
      /// <remarks> Indexing must start at 0 and subsequent class indexes must
      /// increment by 1.<remarks>
      /// </param>
      void register_classes(std::string objects_path);
  }

}
