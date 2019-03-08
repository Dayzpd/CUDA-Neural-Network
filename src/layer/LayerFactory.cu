
#include "Layer.h"
#include "LayerFactory.h"
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"

using namespace std;

#include <stdexcept>

namespace neural_network
{

  class LayerFactory
  {
    public:
      Layer* create(const str layer_type, int num_neurons, Layer* prev_layer=0)
      {
        if (!is_accepted_type(layer_type))
        {
          throw runtime_error(layer_type + " is an invalid layer type. " +
            "Accepted types include: " + this->accepted_types_to_string());
        }

        switch (layer_type)
        {
          case Layer::INPUT: // Input Layer
            InputLayer input_layer;
            input_layer.set_num_neurons(num_neurons);
            input_layer.populate_layer();
            break;
          case Layer::HIDDEN: // Hidden Layer
            HiddenLayer hidden_layer;
            hidden_layer.set_num_neurons(num_neurons);
            hidden_layer.populate_layer();
            hidden_layer.connect
            break;
          case Layer::OUTPUT: // Output Layer
            OutputLayer output_layer;
            break;
        }
      }

      bool is_accepted_type(const str layer_type)
      {
        return find(this->LAYER_TYPES.begin(), this->LAYER_TYPES.end(),
          layer_type);
      }

      string accepted_types_to_string()
      {
        string accepted_types = "[ ";
        for (int x = Layer::INPUT; x < Layer::OUTPUT; x++)
        {
          accepted_types += x + ", ";
        }
        accepted_types += " ]";
        return accepted_types;
      }
  }

}
