#include "Sigmoid.h"

#include <math.h>

#define BLOCK_SIZE 256

namespace neural_network {

  __device__
  float sigmoid(float x)
  {
    return 1.0f / (1 + expf(x));
  }

  __global__
  void device_forward_prop_sigmoid(float* input, float* output
    size_t input_size
  ) {
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread calculates one weights sum + bias.
    if (t_id < input_size && input[t_id] < 0)
    {
      output[t_id] = sigmoid(input[t_id]);
    }
  }

  Sigmoid::Sigmoid()
  {

  }

  Sigmoid::~Sigmoid()
  {

  }

  Neurons& Sigmoid::forward_prop(Neurons& input)
  {
    this->input = input;
    ouput.reserve_memory(input.dim);

    size_t input_size = input.dim.x * input.dim.y;
    int grid_size = ceil(input_size / BLOCK_SIZE);

    device_forward_prop_relu<<<grid_size, BLOCK_SIZE>>>(
      input.device_neurons.get(), output.device_neurons.get() input_size
    );

    return output;
  }

  Neurons& Sigmoid::back_prop(Neurons& input, float learning_rate)
  {

  }

}
