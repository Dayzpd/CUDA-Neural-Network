#include "../layer/LayerFactory.h"
#include "../loss/LossFactory.h"
#include "../optimize/OptimizeFactory.h"
#include "Network.h"


Network::Network(int num_classes, float learning_rate, std::string loss_type,
  std::string optimize_type) : num_classes(num_classes),
  learning_rate(learning_rate)
{
  this->layer_factory = LayerFactory::get_instance();
  this->loss_func = LossFactory::get_instance()->create(loss_type);
  this->optimize_func = OptimizeFactory::get_instance()->create(optimize_type);
}

Network::Network()
{
  for (auto layer : layers)
  {
    delete layer;
  }
}

void Network::add_feature(Neurons feature, Neurons class)
{
  features.push_back(make_tuple(feature, class));
}

void Network::add_layer(std::string layer_type)
{
  this->layers.push_back(this->layer_factory->create(layer_type));
}

void Network::add_layer(std::string layer_type, size_t x, size_t y)
{
  this->layers.push_back(this->layer_factory->create(layer_type, Dim(x, y)));
}

int Network::get_num_classes()
{
  return num_classes;
}

void Network::train()
{
  for (auto feature : features)
  {
    output = forward_propagate(get<0>(feature));
    back_propagate(output, get<1>feature);
  }
}

Neurons& forward_propagate(Neurons& feature)
{
  Neurons layer_output = feature;
  for (auto layer : layers)
  {
    layer_output = layer->forward_prop(layer_output);
  }

  return layer_output;
}

void back_propagate(Neurons& prediction, Neurons& actual)
{

  //TO DO: calculate loss
  Neurons err = this->loss_func->calculate(prediction, actual);

  for (std::vector<int>::reverse_iterator i = layers.rbegin();
    i != layers.rend(); i++)
  {
    err = *i->back_prop(err, learning_rate);
  }
}
