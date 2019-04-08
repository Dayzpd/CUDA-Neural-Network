
#ifndef NETWORK_H
#define NETWORK_H

#include "../loss/LossFunction.h"
#include "../layer/Layer.h"
#include "../optimize/OptimizeFunction.h"

#include <string>
#include <vector>

namespace neural_network
{

  class Network
  {
    private:
      LayerFactory& layer_factory;
      LossFunction& loss_func;
      OptimizeFunction& optimize_func;

      std::vector<Layer*> layers;

      std::vector<Neurons> batches;
      std::vector<Neurons> actuals;

      int num_batches;

    public:
      Network(std::string loss_type, std::string optimize_type);

      ~Network();

      void add_batch(Neurons features, Neurons classes);

      // Activation Layer
      void add_layer(std::string layer_type);

      // Connection Layer
      void add_layer(std::string layer_type, size_t x, size_t y);

      std::string classify(Neurons& prediction);

      int get_num_classes();

      void train();

      void forward_propagate();

      void back_propagate();

      double calculate_loss();
  }

}

#endif
