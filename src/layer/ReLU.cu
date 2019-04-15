#include "ReLU.h"

#include <math.h>

#define BLOCK_SIZE 256

namespace neural_network {

  __global__
  void device_relu(float* input, float* output, size_t input_size)
  {
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (t_id < input_size)
    {
      output[t_id] = fmaxf(input[t_id], 0);
    }
  }

  __global__
  void device_relu_deriv(float* input, float* error, float* delta,
    size_t input_size
  ) {
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (t_id < input_size)
    {
      if (input[t_id] > 0)
      {
        delta[t_id] = error[t_id];
      }
      else
      {
        delta[t_id] = 0;
      }
    }
  }

  ReLU::ReLU() : output(), delta()
  {

  }

  ReLU::~ReLU()
  {

  }

  Neurons& ReLU::forward_prop(Neurons& input)
  {
    this->input = input;
    output.allocate_memory(input.dim);

    int input_size = input.dim.x * input.dim.y;

    // 1D grid of 1D blocks
    dim3 block_size(BLOCK_SIZE);
    dim3 num_blocks(ceil(input_size / BLOCK_SIZE));

    device_relu<<<num_blocks, block_size>>>(
      input.get_device_pointer(), output.get_device_pointer(), input_size
    );

    return output;
  }

  Neurons& ReLU::back_prop(Neurons& error, float learning_rate)
  {
    delta.allocate_memory(input.dim);

    size_t input_size = input.dim.x * input.dim.y;

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(ceil(input_size / BLOCK_SIZE));

    device_relu_deriv<<<grid_size, block_size>>>(input.get_device_pointer(),
      error.get_device_pointer(), delta.get_device_pointer(), input_size);

    return delta;
  }

}
