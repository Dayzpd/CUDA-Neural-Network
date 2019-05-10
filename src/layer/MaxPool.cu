
#include "MaxPool.h"
#include "../neurons/Dim.h"

#include <iostream>
#include <limits>
#include <math.h>
#include <thrust/device_ptr.h>

#define DIM_SIZE_3D 8
#define BLOCK_SIZE 512

__global__
void device_max_pool_forward_prop(float* input, float* max_indices,
  float* output, size_t batch_size, size_t input_dim_size, size_t input_size,
  size_t kernel_dim, size_t output_dim
) {
  int feature_num = blockIdx.x * blockDim.x + threadIdx.x;
  int col_num = blockIdx.y * blockDim.y + threadIdx.y;
  int row_num = blockIdx.z * blockDim.z + threadIdx.z;

  if (feature_num < batch_size && row_num < output_dim && col_num < output_dim)
  {
    // feature_offset = feature_num * input_size
    // row_offset = row_num * input_dim_size
    // Represents the top left corner of the "box" from the input batch for the
    // current feature that we will begin dotting with the kernel values.
    int top_left_index =
      feature_num * input_size + row_num * input_dim_size + col_num;
    float max_val = FLT_MIN;
    int max_index = -1;

    for (int x = 0; x < kernel_dim; x++)
    {
      for (int y = 0; y < kernel_dim; y++)
      {
        if (max_val < input[(int)top_left_index + x * (int)input_dim_size + y])
        {
          max_val = input[(int)top_left_index + x * (int)input_dim_size + y];
          max_index = (int)top_left_index + x * (int)input_dim_size + y;
        }
      }
    }
    int out_index = feature_num * pow((int)output_dim, 2) + row_num * (int)output_dim + col_num;
    max_indices[out_index] = max_index;
    output[out_index] = max_val;
  }
}

__global__
void device_max_pool_back_prop(float* error, float* max_indices, float* delta,
  size_t error_size
) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (x_id < error_size)
  {
    atomicAdd(&delta[(int)max_indices[x_id]], error[x_id]);
  }
}

MaxPool::MaxPool(size_t kernel_dim, std::string name, bool verbose) :
  kernel_dim(kernel_dim), name(name), verbose(verbose)
{

}

MaxPool::~MaxPool()
{
  input.~Neurons();
  output.~Neurons();
  delta.~Neurons();
}

// For testing purposes
Neurons& MaxPool::get_max_indices()
{
  return max_indices;
}


Neurons& MaxPool::forward_prop(Neurons& input)
{
  this->input = input;

  size_t input_dim_size = sqrt(input.dim.y);
  size_t output_dim = input_dim_size - kernel_dim + 1;

  if (max_indices.is_allocated())
  {
    max_indices.zero_device_memory();
  }
  else
  {
    max_indices.allocate_memory(input.dim.x, pow(output_dim, 2));
  }

  output.allocate_memory(input.dim.x, pow(output_dim, 2));

  dim3 block_size(DIM_SIZE_3D, DIM_SIZE_3D, DIM_SIZE_3D);
  dim3 grid_size(
    (input.dim.x + DIM_SIZE_3D - 1) / DIM_SIZE_3D,
    (output_dim + DIM_SIZE_3D - 1) / DIM_SIZE_3D,
    (output_dim + DIM_SIZE_3D - 1) / DIM_SIZE_3D
  );

  if (verbose)
  {
    std::cout << "Block Size: (" << block_size.x << "," << block_size.y << "," << block_size.z << ")" << std::endl;
    std::cout << "Grid Size: (" << grid_size.x << "," << grid_size.y << "," << grid_size.z << ")" << std::endl;
  }

  device_max_pool_forward_prop<<<grid_size, block_size>>>(
    input.get_device_pointer(), max_indices.get_device_pointer(),
    output.get_device_pointer(), input.dim.x, input_dim_size, input.dim.y,
    kernel_dim, output_dim);

  return output;
}

Neurons& MaxPool::back_prop(Neurons& error, float learning_rate)
{
  if (delta.is_allocated())
  {
    delta.zero_device_memory();
  }
  else
  {
    delta.allocate_memory(input.dim);
  }

  dim3 block_size(BLOCK_SIZE);
  dim3 grid_size((error.dim.x * error.dim.y + BLOCK_SIZE - 1) / BLOCK_SIZE);

  device_max_pool_back_prop<<<grid_size, block_size>>>(
    error.get_device_pointer(), max_indices.get_device_pointer(),
    delta.get_device_pointer(), error.dim.x * error.dim.y);

  return delta;
}
