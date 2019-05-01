
#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include "Layer.h"
#include "../neurons/Dim.h"
#include "../neurons/Neurons.h"

namespace cuda_net
{

  class FullyConnected : public Layer
  {
    private:
      Neurons input;
      Neurons weights;
      Neurons biases;
      Neurons output;
      Neurons delta;

    public:
      FullyConnected(Dim dim);

      ~FullyConnected();

      void init_weights();

      void init_biases();

      Neurons& forward_prop(Neurons& input);

      Neurons& back_prop(Neurons& error, float learning_rate);

      void backprop_error(Neurons& error);

      void update_weights(Neurons& error, float learning_rate);

      void update_bias(Neurons& error, float learning_rate);
  };

}

#endif
