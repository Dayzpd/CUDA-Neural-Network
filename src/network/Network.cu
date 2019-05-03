
#include "Network.h"

namespace cuda_net
{

  Network::Network()
  {

  }

  Network::~Network()
  {
    for (auto layer : layers)
    {
      delete layer;
    }
  }

  void Network::add_layer(Layer* layer)
  {
    layers.push_back(layer);
  }

  Neurons Network::forward_propagate(Neurons& feature)
  {
    Neurons output = feature;
    for (auto layer : layers)
    {
      output = layer->forward_prop(output);
    }

    prob = output;

    return prob;
  }

  void Network::back_propagate(Neurons& prob, Neurons& actual,
    float learning_rate
  ) {
    prob_delta.allocate_memory(prob.dim);

    Neurons err = loss_func.calculate_deriv(prob, actual, prob_delta);

    for (auto iter = layers.rbegin(); iter != layers.rend(); iter++)
    {
      err = (*iter)->back_prop(err, learning_rate);
    }

  }

  float Network::batch_train_loss(Neurons& prob, Neurons& batch_targets)
  {
    return loss_func.calculate(prob, batch_targets);
  }

}
