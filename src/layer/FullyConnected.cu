
#include "FullyConnected.h"

#include <assert.h>
#include <math.h>
#include <random>

#define DIM_SIZE 16
#define BLOCK_SIZE 256

namespace neural_network {

  __global__
  void device_forward_prop_fc(float* input, float* weights, float* biases,
    float* output, size_t input_x, size_t weights_x, size_t weights_y
  ) {
    int x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int y_id = blockIdx.y * blockDim.y + threadIdx.y;

    // (x_id, y_id) refer to the (row, col) of the output
    if (x_id < input_x && y_id < weights_y)
    {
      int input_row = weights_x * x_id; // input_row: maps input values
      int weights_col = weights_x * y_id; // weights_col: maps input values
      int out_id = weights_y * x_id + y_id; // out_id: maps output values

      float output_val = 0;

      // output = sum(inputs * weights) + bias
      for (int i = 0; i < weights_x; i++)
      {
        output_val += input[input_row + i] * weights[weights_col + i];
      }

      output[out_id] = output_val + biases[y_id];
    }
  }

  FullyConnected::FullyConnected(Dim dim) : weights(dim.x, dim.y),
    biases(1, dim.y), output(), backprop_deriv()
  {
    init_weights();
    init_biases();
  }

  void FullyConnected::init_weights()
  {
    std::default_random_engine rand;
    std::normal_distribution<float> norm(0.0, 1.0);

    for (size_t x = 0; x < weights.dim.x; x++)
    {
      for (size_t y = 0; y < weights.dim.y; y++)
      {
        weights.host[x + weights.dim.x * y] = norm(rand);
      }
    }

    weights.memcpy_host_to_device();
  }

  void FullyConnected::init_biases()
  {
    for (size_t y = 0; y < weights.dim.y; y++)
    {
      biases.host[y] = 0.01;
    }

    biases.memcpy_host_to_device();
  }

  Neurons& FullyConnected::forward_prop(Neurons& input)
  {
    assert(input.dim.y == weights.dim.x);

    this->input = input;
    this->output.allocate_memory(Dim(input.dim.x, weights.dim.y));

    // 1D grid of 2D blocks
    dim3 block_size(DIM_SIZE, DIM_SIZE);
    dim3 num_blocks(ceil(input.dim.x * weights.dim.y / (BLOCK_SIZE)));

    device_forward_prop_fc<<<ceil(num_blocks, block_size>>>(
      input.get_device_pointer(), weights.get_device_pointer(),
      biases.get_device_pointer(), output.get_device_pointer(),
      input.dim.x, weights.dim.x, weights.dim.y);

    return output;
  }

  Neurons& FullyConnected::back_prop(Neurons& input, float learning_rate)
  {

  }


}
