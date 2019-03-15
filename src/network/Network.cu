
#include "Layer.h"
#include "LayerFactory.h"
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"

#include <memory>
#include <stdexcept>

namespace neural_network
{

  Network::Network()
  {
    this->num_layers = 0;
  }

  Network::~Network()
  {
    // delete layers: start with head and continue thru to tail.
  }

  void Network::add_layer()
  {

  }



}
