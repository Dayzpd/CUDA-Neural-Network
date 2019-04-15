#include "../layer/LayerFactory.h"
#include "../loss/LossFactory.h"
#include "../optimize/OptimizeFactory.h"
#include "Network.h"


Network::Network(std::string loss_type, std::string optimize_type) :
  num_batches(0), prob(), prob_delta()
{
  this->layer_factory = LayerFactory::get_instance();
  this->loss_func = LossFactory::get_instance()->create(loss_type);
  this->optimize_func = OptimizeFactory::get_instance()->create(optimize_type);
}

Network::~Network()
{
  for (auto layer : layers)
  {
    delete layer;
  }

  for (auto feature : features)
  {
    delete feature;
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
  this->layers.push_back(this->layer_factory->create(layer_type));
}

void Network::add_layer(std::string layer_type, size_t x, size_t y)
{
  this->layers.push_back(this->layer_factory->create(layer_type,
    Dim(x, y)));
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

  return this->classes[max];
}

int Network::get_num_classes()
{
  return this->classes.size();
}

void Network::train(int num_epochs, float learning_rate, int checkpt = 100)
{
  for (int e = 0; e < num_epochs; e++)
  {
    float loss = 0.0;
    for (int b = 0; b < num_batches; b++)
    {
      prob = forward_propagate(batches.at(b));
      back_propagate(prob, batches.at(b), learning_rate);
      float loss += this->loss_func.calculate(output, actuals.at(b));
    }

    if (e % checkpt == 0)
    {
      printf("Epoch: %d, Cost: %f", e, loss / num_batches);
    }
  }
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
