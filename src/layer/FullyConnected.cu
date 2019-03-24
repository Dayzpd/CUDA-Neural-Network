
#include "FullyConnected.h"

#include <numeric>

namespace neural_network {

  FullyConnected::FullyConnected(int n_neurons, std::string act_type,
    Layer& p_layer
  ) {
    this->num_neurons = n_neurons;
    this->set_activation_function(act_type);
    this->prev_layer = p_layer;
    this->input_size = this->prev_layer.get_num_outputs();
    this->build_neurons();
    this->prev_layer.set_next_layer(*this);
  }


  FullyConnected::FullyConnected(int n_neurons, std::string act_type,
    int in_size
  ) {
    this->num_neurons = n_neurons;
    this->set_activation_function(act_type);
    this->prev_layer = NULL;
    this->input_size = in_size;
    this->build_neurons();
  }

  void FullyConnected::build_neurons()
  {
    this->weights = std::vector<double>(num_neurons);
    this->biases = std::vector<double>(num_neurons);

    for (int x = 0; x < this->num_neurons * this->input_size; x++)
    {
      this->weights[x] = this->rand_norm();
      this->biases[x] = this->rand_norm();
    }
  }

  const int& FullyConnected::get_num_outputs()
  {
    return this->num_neurons;
  }

  const std::vector<double>& FullyConnected::activate(
    const std::vector<double>& input
  ) {
    std::vector<double> output(this->num_neurons);

    for (int x = 0; x < this->num_neurons; x++)
    {
      auto first = this->weights.begin() + x * this->num_inputs;
      auto last = this->weights.begin() + (x + 1) * this->num_inputs - 1;
      output[x] += std::inner_product(first, last, input, 0.0)
        + this->biases.at(x);
    }
  }

}
