
#include "Network.h"
#include "../layer/LayerFactory.h"
#include "../loss/LossFactory.h"

namespace cuda_net
{

  Network::Network(std::string loss_type) :
    loss_func(LossFactory::get_instance().create(loss_type))
  {

  }

  Network::~Network()
  {
    for (auto& layer : layers)
    {
      layer.reset();
    }
  }

  void Network::add_layer(std::string layer_type)
  {
    layers.push_back(LayerFactory::get_instance().create(layer_type));
  }

  void Network::add_layer(std::string layer_type, size_t x, size_t y)
  {
    layers.push_back(LayerFactory::get_instance().create(layer_type, Dim(x, y)));
  }

  Neurons Network::forward_propagate(Neurons& feature)
  {
    Neurons output = feature;
    for (auto& layer : layers)
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

    Neurons err = loss_func->calculate_deriv(prob, actual, prob_delta);

    for (auto iter = layers.rbegin(); iter != layers.rend(); iter++)
    {
      err = (*iter)->back_prop(err, learning_rate);
    }

  }

}
