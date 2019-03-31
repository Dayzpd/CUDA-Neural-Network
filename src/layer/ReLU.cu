#include "ReLU.h"

#include <math.h>

#define BLOCK_SIZE 256

namespace neural_network {

  __global__
  void device_forward_prop_relu(float* input, size_t input_size)
  {
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread calculates one weights sum + bias.
    if (t_id < input_size && input[t_id] < 0)
    {
      input[t_id] = 0;
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
    size_t input_size = input.dim.x * input.dim.y;
    int grid_size = ceil(input_size / BLOCK_SIZE);

    device_forward_prop_relu<<<grid_size, BLOCK_SIZE>>>(
      input.device_neurons.get(), input_size
    );

    return input;
  }

  Neurons& ReLU::back_prop(Neurons& input, float learning_rate)
  {

  }

}
