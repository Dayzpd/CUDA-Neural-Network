#include "../layer/LayerFactory.h"
#include "../loss/LossFactory.h"
#include "../optimize/OptimizeFactory.h"
#include "Network.h"

namespace neural_network
{

  Network::Network(std::string loss_type) : prob(), prob_delta()
  {

  }

  Network::~Network()
  {
    for (auto layer : layers)
    {
      delete layer;
    }
  }

  void Network::add_batch(Neurons features, Neurons classes)
  {
    batches.push_back(features);
    actuals.push_back(classes);
    num_batches += 1;
  }

  void Network::add_layer(std::string layer_type)
  {
    layers.push_back(LayerFactory::get_instance().create(layer_type));
  }

  void Network::add_layer(std::string layer_type, size_t x, size_t y)
  {
    layers.push_back(LayerFactory::get_instance().create(layer_type, Dim(x, y)));
  }

  std::string classify(Neurons& prediction)
  {
    prediction->memcpy_device_to_host();

    float max = prediction[0];
    for (size_t x = 1; x < prediction.dim.x * prediction.dim.y; x++)
    {
      if (max < prediction[x])
      {
        max = prediction[x];
      }
    }

    return max;
  }

  void Network::train(int num_epochs, float learning_rate, int checkpt = 100)
  {

  }

  Neurons forward_propagate(Neurons& feature)
  {
    Neurons output = feature;
    for (auto layer : layers)
    {
      output = layer->forward_prop(layer_output);
    }

    prob = output;

    return prob;
  }

  void back_propagate(Neurons& prob, Neurons& actual)
  {
    prob_delta.allocate_memory(prob.dim);

    Neurons err = this->loss_func->calculate_deriv(prob, actual, prob_delta);

    for (std::vector<int>::reverse_iterator i = layers.rbegin();
      i != layers.rend(); i++)
    {
      err = *i->back_prop(err, learning_rate);
    }
  }

}
