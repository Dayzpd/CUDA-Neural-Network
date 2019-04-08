
#include "FullyConnected.h"

#include <assert.h>
#include <math.h>
#include <random>

#define BLOCK_DIM_SIZE 16
#define BLOCK_SIZE 256

namespace neural_network {

  __global__
  void device_forward_prop_fc(float* input, float* weights, float* biases,
    float* output, size_t input_x, size_t weights_x, size_t weights_y
  ) {
    // x_id used to access input values
    int x_id = blockIdx.x * blockDim.x + threadIdx.x;
    // y_id used to access weight values
    int y_id = blockIdx.y * blockDim.y + threadIdx.y;

    // (x_id, y_id) refer to the (row, col) of the output
    if (x_id < input_x && y_id < weights_y)
    {
      // t_id used to assign output values
      int t_id = x_id + input.dim.x * y_id;

      float output_val = 0;

      // output = sum(inputs * weights) + bias
      for (int i = 0; i < weights_x; i++)
      {
        output_val +=
          input[x_id + (i * input_x)] * weights[i + (y_id * weights_x)];
      }

      output[t_id] = output_val + biases[y_id];
    }
  }

  FullyConnected::FullyConnected(Dim dim) : weights(dim.x, dim.y),
    biases(1, dim.y), layer_output(1, dim.y)
  {
    weights.reserve_memory();
    biases.reserve_memory();
    layer_output.reserve_memory();
    initialize_neurons();
  }

  void FullyConnected::initialize_neurons()
  {
    init_weights();
    init_biases();
    init_layer_ouput();
  }

  void FullyConnected::init_weights()
  {
    std::default_random_engine rand;
    std::normal_distribution<float> norm(0.0, 1.0);

    for (size_t x = 0; x < weights.dim.x; x++)
    {
      for (size_t y = 0; y < weights.dim.y; y++)
      {
        weights[x + weights.dim.x * y] = norm(rand);
      }
    }

    weights.memcpy_host_to_device();
  }

  void FullyConnected::init_biases()
  {
    for (size_t y = 0; y < weights.dim.y; y++)
    {
      biases[y] = 0.01;
    }

    biases.memcpy_host_to_device();
  }

  Neurons& FullyConnected::forward_prop(Neurons& input)
  {
    assert(input.dim.y == weights.dim.x);

    this->input = input;
    this->output.reserve_memory(Dim(input.dim.x, weights.dim.y));

    // 1D grid of 2D blocks
    dim3 block_size(BLOCK_DIM_SIZE, BLOCK_DIM_SIZE);
    dim3 num_blocks(ceil(input.dim.x * weights.dim.y / (BLOCK_SIZE)));

    device_forward_prop_fc<<<ceil(num_blocks, block_size>>>(
      input.device_neurons.get(), weights.device_neurons.get(),
      biases.device_neurons.get(), output.device_neurons.get(),
      input.dim.x, weights.dim.x, weights.dim.y);

    return output;
  }

  Neurons& FullyConnected::back_prop(Neurons& input, float learning_rate)
  {

  }


}
