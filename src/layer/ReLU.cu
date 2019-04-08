#include "ReLU.h"

#include <math.h>

#define BLOCK_SIZE 256

namespace neural_network {

  __global__
  void device_forward_prop_relu(float* input, float* output, size_t input_size)
  {
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (t_id < input_size)
    {
      output[t_id] = fmaxf(input[t_id], 0);
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

    int input_size = input.dim.x * input.dim.y;

    // 1D grid of 1D blocks
    dim3 block_size(BLOCK_SIZE);
    dim3 num_blocks(ceil(input_size / BLOCK_SIZE));

    device_forward_prop_relu<<<num_blocks, block_size>>>(
      input.device_neurons.get(), output.device_neurons.get(), input_size
    );

    return output;
  }

  Neurons& ReLU::back_prop(Neurons& input, float learning_rate)
  {

  }

}
