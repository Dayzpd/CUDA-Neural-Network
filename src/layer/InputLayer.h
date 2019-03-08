using namespace std;

#include "Layer.h"

class InputLayer : public Layer
{
  private:
    int num_input_neurons;
    vector<InputNeuron> neurons;

  public:
    const string LAYER_TYPE = Layer::ALLOWED_LAYER_TYPES[0];
}
