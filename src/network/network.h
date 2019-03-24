
#ifndef LAYER_H
#define LAYER_H

#include "../loss/LossFunction.h"
#include "../layer/Layer.h"
#include "../optimize/OptimizeFunction.h"

#include <tuple>
#include <vector>

namespace neural_network
{

  class Network
  {
    private:
      int num_classes;
      int learning_rate;

      LossFunction* loss;
      OptimizeFunction* optimize;

      Layer* first_layer;
      Layer* output_layer;

    public:
      Network(int n_classes, int lr, std::string loss_type,
        std::string opt_type);

      ~Network();

      void add_layer(Layer* Layer);

      void train();

  }

}

#endif
