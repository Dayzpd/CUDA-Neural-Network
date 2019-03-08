using namespace std;

#include "Layer.h"

class HiddenLayer : public Layer
{
  private:
    int num_hidden_neurons;
    vector<HiddenNeuron> neurons;

  public:
    const string LAYER_TYPE = Layer::ALLOWED_LAYER_TYPES[1];
}
