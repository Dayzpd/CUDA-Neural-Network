
#ifndef NETWORK_H
#define NETWORK_H

#include "../loss/CrossEntropy.h"
#include "../layer/Layer.h"
#include "../neurons/Neurons.h"

#include <memory>
#include <string>
#include <vector>

namespace cuda_net
{

  class Network
  {
    private:
      std::vector<Layer*> layers;

      Neurons prob;
      Neurons prob_delta;

      CrossEntropy loss_func;

    public:
      Network();

      ~Network();

      void add_layer(Layer* layer);

      void train();

      Neurons forward_propagate(Neurons& feature);

      void back_propagate(Neurons& prob, Neurons& actual, float learning_rate);

      float batch_train_loss(Neurons& prob, Neurons& batch_targets);
  };

}

#endif
