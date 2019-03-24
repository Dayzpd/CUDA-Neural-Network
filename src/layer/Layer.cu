
#include "../activation/ActivationFactory.h"
#include "Layer.h"

#include <stdlib.h>
#include <vector>

namespace neural_network
{

  double Layer::rand_norm()
  {
    return ((double) rand() / 1);
  }

  const int& Layer::get_num_neurons()
  {
    return *this->num_neurons;
  }

  void set_activation_function(string act_type)
  {
    this->act_func = ActivationFactory::create(act_type);
  }

  void set_prev_layer(Layer& p_layer)
  {
    this->prev_layer = p_layer;
  }

  void set_next_layer(Layer& n_layer)
  {
    this->next_layer = n_layer;
  }

}
