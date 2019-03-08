using namespace std;

#include "Layer.h"

class OutputLayer : public Layer
{
  private:
    int num_output_classes;
    vector<OutputNeuron> neurons;

  public:
    const string LAYER_TYPE = Layer::ALLOWED_LAYER_TYPES[2];
}
