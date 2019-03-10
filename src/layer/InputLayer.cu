#include "Layer.h"
#include "InputLayer.h"

namespace neural_network {

  class InputLayer : public Layer
  {
    private:
      int num_input_neurons;
      vector<InputNeuron> neurons;

    public:
      InputLayer();

      ~InputLayer();

      /// <summary>Adds a new neuron to this <c>InputLayer</c> instance.
      /// </summary>
      /// <param name="neuron">The new <c>InputNeuron</c> to be added.
      /// </param name>
      /// <returns>None</returns>
      void add_neuron(InputNeuron neuron)
      {
        this->neurons.push_back(neuron);
      }

      /// <summary>Populates this <c>InputLayer</c> instance with as many
      /// neurons specified using the <c>set_num_neurons</c> method.</summary>
      /// <returns>None.</returns>
      void populate_layer()
      {
        for (int x = 0; x < this->num_input_neurons; x++)
        {
          InputNeuron neuron;
          this->add_neuron(neuron);
        }
      }

      /// <summary>Returns the number of <c>InputNeuron</c> objects in this
      /// <c>InputLayer</c>.</summary>
      /// <returns>Returns number of neurons in the <c>InputLayer</c> as an
      /// integer.</returns>
      void get_num_neurons()
      {
        return this->num_input_neurons;
      }

      /// <summary>Sets how many neurons should be in this <c>InputNeuron</c>.
      /// </summary>
      /// <param name="num_neurons">Specifies how many neurons should be in this
      /// <c>InputLayer</c>.</param name>
      /// <returns>None</returns>
      void set_num_neurons(int num_neurons)
      {
        this->num_input_neurons = num_neurons;
      }
  }

}
