
namespace neural_network
{

  Network::Network(int input_size, std::vector<string> classes,
    double learning_rate, LossFunction::Type loss_strategy,
    OptimizeFunction::Type optimize_function)
  {
    this->input_size = input_size;
    this->classes = classes;
    this->learning_rate = learning_rate;
    this->loss_function = LossFunction::create(loss_strategy);
    this->optimize_function = OptimizeFunction::create(optimize_strategy);
    this->head_layer = NULL;
    this->tail_layer = NULL;
  }

  Network::~Network()
  {
    // delete layers: start with head and continue thru to tail.
  }

  void Network::add_layer()
  {
     = new Neuron(num_neurons, Layer::create(layer_strategy),
      ActivationFunction::create(activation_strategy));

    if (head_layer == NULL)
    {
      layer->set_num_connections
    }
    else
    {

    }
  }



}
