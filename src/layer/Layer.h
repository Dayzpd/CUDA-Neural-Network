
#ifndef LAYER_H
#define LAYER_H

#include "../activation/ActivationFunction.h"
#include "../connections/Connection.h"
#include "../neuron/Neurons.h"


namespace neural_network
{

  class Layer
  {
    private:
      int input_size;
      int num_neurons;
      int num_biases;
      // This will likely need to change(triplet/tuple?). Fur the purpose of the project,
      // it will likely be limited to 3 dims.
      int input_dimensions;
      int dimension_length;

      Neurons* weights;
      Neurons* biases;

      ActivationFunction* act_func;
      Connection* connection;

      Layer* prev_layer;
      Layer* next_layer;

    public:
      Layer(int input_size, int num_neurons, int num_biases,
        std::string connection_type, std::string activation_type,
        Layer* prev_layer=NULL);

      ~Layer();

      void build_layer(); // May end up returning Layer* and this class may become
                          // LayerBuilder.

      int get_input_size();

      void set_input_size(int in_size);

      int get_num_neurons();

      void set_num_neurons(int n_neurons);

      int get_num_biases();

      void set_num_biases(int n_biases);

      ActivationFunction* get_act_func();

      void set_activation_function(string activation_Type);

      Layer* get_prev_layer();

      void set_prev_layer(Layer* p_layer);

      Layer* get_next_layer();

      void set_next_layer(Layer* n_layer);
  }

}

#endif
