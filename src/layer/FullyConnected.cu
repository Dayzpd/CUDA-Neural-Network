
#include "FullyConnected.h"

#include <assert.h>
#include <math.h>
#include <random>

#define BLOCK_SIZE 256

namespace neural_network {

  __global__
  void device_forward_prop_fc(float* weights, float* biases, float* layer_input,
    float* layout_ouput, size_t weights_x, size_t weights_y
  ) {
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread calculates one weights sum + bias.
    if (t_id < weights_y)
    {
      float output = 0;

      // sum(input[i] * weight[i]) + bias[i]
      for (int index = 0; index < weights_x; index++)
      {
        output += weights[t_id * weights_x + index] * layer_input[index];
      }

      layer_output[t_id] = output + biases[t_id];
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

  void FullyConnected::init_layer_ouput()
  {
    for (size_t y = 0; y < layer_ouput.dim.y; x++)
    {
      layer_ouput[y] = 0;
    }

    layer_ouput.memcpy_host_to_device();
  }

  void FullyConnected::init_optimized()
  {
    for (size_t x = 0; x < weights.dim.x; x++)
    {
      optimized[x] = 0;
    }

    optimized.memcpy_host_to_device();
  }

  Neurons& FullyConnected::forward_prop(Neurons& input)
  {
    // In order to multiply two matrices in a Fully Connected layer, weights
    // x dim, must equal the input x dim * y dim.
    assert((input.dim.x * input.dim.y) == weights.dim.x);

    device_forward_prop_fc<<<ceil(weights.dim.x / BLOCK_SIZE), BLOCK_SIZE>>>(
      weights.device_neurons.get(), biases.device_neurons.get(),
      layer_input.device_neurons.get(), layout_ouput.device_neurons.get(),
      weights.dim.x, weights.dim.y)

    return layer_ouput;
  }

  Neurons& FullyConnected::back_prop(Neurons& input, float learning_rate)
  {

  }


}
