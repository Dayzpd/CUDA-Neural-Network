#include "ReLU.h"

#include <math.h>

#define BLOCK_SIZE 256

namespace neural_network {

  __global__
  void device_forward_prop_relu(float* input, float* output, size_t input_size)
  {
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread calculates one weights sum + bias.
    if (t_id < input_size)
    {
      if (input[t_id] < 0)
      {
        output[t_id] = 0;
      }
      else
      {
        output[t_id] = input[t_id];
      }
    }
  }

  ReLU::ReLU()
  {

  }

  ReLU::~ReLU()
  {

  }

  Neurons& ReLU::forward_prop(Neurons& input)
  {
    this->input = input;
    output.reserve_memory(input.dim);

    size_t input_size = input.dim.x * input.dim.y;
    int grid_size = ceil(input_size / BLOCK_SIZE);

    device_forward_prop_relu<<<grid_size, BLOCK_SIZE>>>(
      input.device_neurons.get(), output.device_neurons.get(), input_size
    );

    return output;
  }

  Neurons& ReLU::back_prop(Neurons& input, float learning_rate)
  {

  }

}
