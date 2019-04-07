#include "Softmax.h"

#include <math.h>

#define BLOCK_SIZE 256

namespace neural_network {

  __device__
  float sigmoid(float x)
  {
    return 1.0f / (1 + expf(x));
  }

  __global__
  void device_softmax(float* input, float* output, size_t input_size)
  {
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread calculates one weights sum + bias.
    /*if (t_id < input_size && input[t_id] < 0)
    {
      output[t_id] = sigmoid(input[t_id]);
    }*/
    /*
    1. find max value: max = max(input)
    2. input_modified[i] = expf(input[i] - max)
    3. find sum of modified input vals: sum = sum(input_modified)
    4. output[x] = input_modified[x] / sum 
    */
  }

  Softmax::Softmax()
  {

  }

  Softmax::~Softmax()
  {

  }

  Neurons& Softmax::forward_prop(Neurons& input)
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

  Neurons& Softmax::back_prop(Neurons& input, float learning_rate)
  {

  }

}
