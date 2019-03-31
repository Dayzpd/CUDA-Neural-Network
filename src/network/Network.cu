#include "LayerFactory.h"
#include "Network.h"


Network::Network()
{
  this->layer_factory = LayerFactory::get_instance();
}

Network::Network()
{
  for (auto layer : layers)
  {
    delete layer;
  }
}

Network::add_layer(std::string layer_type)
{
  this->layers.push_back(this->layer_factory(layer_type));
}

Network::add_layer(std::string layer_type, size_t x, size_t y)
{
  this->layers.push_back(this->layer_factory(layer_type, Dim(x, y)));
}

void Network::train(Features& features, int batch_size, float learning_rate)
{

}

Neurons& forward_propagate(Neurons& feature)
{

}

void back_propagate(Neurons& prediction, Neurons& actual)
{

}
