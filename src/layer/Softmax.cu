#include "Softmax.h"

#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>

#define DIM_SIZE 32
#define BLOCK_SIZE 1024

namespace neural_network {

  __global__
  void device_row_max_values(float* input, float* max_vals, size_t input_x,
    size_t input_y
  ) {
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_id < input_x)
    {
      for (size_t col = 0; col < input_y; col++)
      {
        if (max_vals[row_id] < input[row_id * input_y + col])
        {
          max_vals[row_id] = input[row_id * input_y + col];
        }
      }
    }
  }

  __global__
  void device_softmax(float* input, float* max_vals, float* row_sums,
    float* output, size_t input_x, size_t input_y
  ) {
    int x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int y_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_id < input_x && y_id < input_y)
    {
      int t_id = x_id * input_y + y_id;

      output[t_id] = expf(input[t_id] - max_vals[x_id]);
      atomicAdd(&row_sums[x_id], output[tid]);
      __syncthreads();

      output[t_id] = output[t_id] / row_sums[x_id]
    }
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
    output.allocate_memory(input.dim);

    float* row_max = nullptr;
    cudaMalloc((void**)&row_max, input.dim.x * sizeof(float));
    cudaMemset(row_max, -1.0f, input.dim.x * sizeof(float));

    float* row_sums = nullptr;
    cudaMalloc((void**)&row_sums, input.dim.x * sizeof(float));
    cudaMemset(row_sums, 0, input.dim.x * sizeof(float));

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(ceil(input.dim.x / BLOCK_SIZE));

    device_max_row_values<<<grid_size, block_size>>>(input.get_device_pointer(),
      &row_max[0], input.dim.x, input.dim.y);

    dim3 block_size(DIM_SIZE, DIM_SIZE - (DIM_SIZE % input.dim.y));
    dim3 grid_size(ceil((input.dim.x * input.dim.y)
      / (block_size.x * block_size.y)));

    device_softmax<<<grid_size, block_size>>>(input.get_device_pointer(),
      &row_max[0], &row_sums[0], output.get_device_pointer(), input.dim.x,
      input.dim.y);

    cudaFree(row_max);
    cudaFree(row_sums);

    return output;
  }

  Neurons& Softmax::back_prop(Neurons& input, float learning_rate)
  {

  }

}
