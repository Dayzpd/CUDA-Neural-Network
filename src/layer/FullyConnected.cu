
#include "FullyConnected.h"

#include <assert.h>
#include <iostream>
#include <math.h>
#include <random>

#include <thrust/system_error.h>

#define DIM_SIZE 32
#define BLOCK_SIZE 1024

__global__
void device_forward_prop_fc(float* input, float* weights, float* biases,
  float* output, size_t input_x, size_t weights_x, size_t weights_y,
  bool verbose
) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int y_id = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_id < input_x && y_id < weights_y)
  {
    int input_row = weights_x * x_id; // input_row: maps input values
    int weights_col = weights_x * y_id; // weights_col: maps weights values
    int out_id = weights_y * x_id + y_id; // out_id: maps output values

    float output_val = 0;

    // output = sum(inputs * weights) + bias
    for (int i = 0; i < weights_x; i++)
    {
  	   output_val += input[input_row + i] * weights[weights_col + i];
    }

    output[out_id] = output_val + biases[y_id];

    if (verbose)
    {
      printf("out[%d] = %f\n", out_id, output_val);
    }
  }
}

__global__
void device_backprop_error(float* error, float* weights, float* delta,
  size_t error_x, size_t error_y, size_t weights_x
) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int y_id = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_id < error_x && y_id < weights_x)
  {
    int err_off = x_id * error_y;
    int weight_off = y_id * error_y;

    float delta_val = 0.0f;
    for (size_t i = 0; i < error_y; i++)
    {
  	   delta_val += error[err_off + i] * weights[weight_off + i];
    }

    delta[x_id * weights_x + y_id] = delta_val;
  }
}

__global__
void device_update_weights(float* input, float* error, float* weights,
  size_t input_x, size_t input_y, size_t error_y, float learning_rate
) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x; // used to access input
  int y_id = blockIdx.y * blockDim.y + threadIdx.y; // used to access error

  if (x_id < input_y && y_id < error_y)
  {
    float update_val = 0.0f;
    for (size_t i = 0; i < input_x; i++)
    {
  	   update_val += error[y_id + error_y * i] * input[x_id + input_y * i];
    }

    weights[y_id * input_y + x_id] -= learning_rate * (update_val / input_x);
  }
}

__global__
void device_update_bias(float* error, float* biases, size_t error_x,
  size_t error_y, float learning_rate
) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x; // error row
  int y_id = blockIdx.y * blockDim.y + threadIdx.y; // error col & bias index

  if (x_id < error_x && y_id < error_y)
  {
    atomicAdd(&biases[y_id],
  	   - learning_rate * (error[x_id * error_y + y_id] / error_x));
  }
}

FullyConnected::FullyConnected(Dim dim, std::string name, bool verbose) :
  weights(dim.x, dim.y), biases(1, dim.y), name(name), verbose(verbose)
{
  weights.allocate_memory();
  init_weights();
  biases.allocate_memory();
  init_biases();
}

FullyConnected::~FullyConnected()
{
  weights.~Neurons();
  biases.~Neurons();
}

void FullyConnected::init_weights()
{
  std::default_random_engine rand;
  std::normal_distribution<float> norm(
    0.f,
    1.f / sqrt((float)weights.dim.x)
  );

  size_t total_len = weights.dim.x * weights.dim.y;

  for (size_t x = 0; x < total_len; x++)
  {
    weights.host_data[x] = norm(rand);

    if (verbose)
    {
      printf("weights[%d]: %f\n", (int)(x), weights.host_data[x]);
    }
  }
  weights.memcpy_host_to_device();
}

void FullyConnected::init_biases()
{
  for (size_t y = 0; y < weights.dim.y; y++)
  {
    biases.host_data[y] = 0.01;
  }

  biases.memcpy_host_to_device();
}

Neurons& FullyConnected::forward_prop(Neurons& input)
{
  assert(input.dim.y == weights.dim.x);

  this->input = input;
  output.allocate_memory(Dim(input.dim.x, weights.dim.y));

  // 2D grid of 2D blocks
  dim3 block_size(DIM_SIZE, DIM_SIZE);
  dim3 grid_size(
    (input.dim.x + DIM_SIZE - 1) / DIM_SIZE,
    (weights.dim.y + DIM_SIZE - 1) / DIM_SIZE
  );

  device_forward_prop_fc<<<grid_size, block_size>>>(
    input.get_device_pointer(), weights.get_device_pointer(),
    biases.get_device_pointer(), output.get_device_pointer(),
    input.dim.x, weights.dim.x, weights.dim.y, false);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    std::cerr << "Error (FullyConnected::forward_prop[" << __LINE__ <<
      "]): Kernel launch failed.\n" <<
      "Block Size: (" << block_size.x << ", " << block_size.y << ")\n" <<
      "Grid Size: (" << grid_size.x << ", " << grid_size.y << ")\n" <<
      "CUDA Error Message: " <<
      cudaGetErrorString(err) << std::endl;
    exit(-1);
  }

  return output;
}

Neurons& FullyConnected::back_prop(Neurons& error, float learning_rate)
{
  delta.allocate_memory(input.dim);

  backprop_error(error);

  update_weights(error, learning_rate);

  update_bias(error, learning_rate);

  return delta;
}

void FullyConnected::backprop_error(Neurons& error)
{
  dim3 block_size(DIM_SIZE, DIM_SIZE);
  dim3 grid_size(
    (delta.dim.x + DIM_SIZE - 1) / DIM_SIZE,
    (delta.dim.y + DIM_SIZE - 1) / DIM_SIZE
  );

  device_backprop_error<<<grid_size, block_size>>>(error.get_device_pointer(),
    weights.get_device_pointer(), delta.get_device_pointer(), error.dim.x,
    error.dim.y, weights.dim.x);
}

void FullyConnected::update_weights(Neurons& error, float learning_rate)
{
  assert(error.dim.x == input.dim.x);
  assert(error.dim.y == weights.dim.y);
  assert(input.dim.y == weights.dim.x);

  dim3 block_size(DIM_SIZE, DIM_SIZE);
  dim3 grid_size(
    (weights.dim.x + DIM_SIZE - 1) / DIM_SIZE,
    (weights.dim.y + DIM_SIZE - 1) / DIM_SIZE
  );

  device_update_weights<<<grid_size, block_size>>>(input.get_device_pointer(),
    error.get_device_pointer(), weights.get_device_pointer(), input.dim.x,
    input.dim.y, error.dim.y, learning_rate);
}

void FullyConnected::update_bias(Neurons& error, float learning_rate)
{
  assert(error.dim.y == biases.dim.y);

  dim3 block_size(DIM_SIZE, DIM_SIZE);
  dim3 grid_size(
    (error.dim.x + DIM_SIZE - 1) / DIM_SIZE,
    (error.dim.y + DIM_SIZE - 1) / DIM_SIZE
  );

  device_update_bias<<<grid_size, block_size>>>(error.get_device_pointer(),
    biases.get_device_pointer(), error.dim.x, error.dim.y, learning_rate);
}
