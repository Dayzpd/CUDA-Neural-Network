#include "Softmax.h"

#include <math.h>

#define DIM_SIZE_2D 32
#define DIM_SIZE_3D 8
#define BLOCK_SIZE 1024

namespace cuda_net
{

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
    int t_id = x_id * input_y + y_id;

    if (x_id < input_x && y_id < input_y)
    {
      output[t_id] = expf(input[t_id] - max_vals[x_id]);
      atomicAdd(&row_sums[x_id], output[t_id]);
      __syncthreads();

      output[t_id] = output[t_id] / row_sums[x_id];
    }
  }

  __global__
  void device_softmax_jacobian(float* softmax, float* jacobian,
    size_t softmax_x, size_t softmax_y, size_t jacobian_y
  ) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // x_id: feature row
    int i = blockIdx.y * blockDim.y + threadIdx.y; // y_id: i offset
    int j = blockIdx.z * blockDim.z + threadIdx.z; // z_id: j offset

    if (row < softmax_x && i < softmax_y && j < softmax_y)
    {
      int j_ind = row * jacobian_y + i * softmax_y + j; // jacobian index
      int s_ind = row * softmax_y; // softmax offset
      int kronecker = (i == j) ? 1.0f : 0.0f;

      jacobian[j_ind] = softmax[s_ind + i] * (kronecker - softmax[s_ind + j]);
    }
  }

  __global__
  void device_softmax_error(float* error, float* jacobian, float* delta,
    size_t softmax_x, size_t softmax_y, size_t jacobian_y
  ) {
    int x_id = blockIdx.x * blockDim.x + threadIdx.x; // x_id: row #
    int y_id = blockIdx.y * blockDim.y + threadIdx.y; // y_id: col #

    if (x_id < softmax_x && y_id < softmax_y)
    {
      int delta_val = 0.0f;
      int err_off = x_id * softmax_y; // error offset
      int jacob_off = x_id * jacobian_y + y_id * softmax_y; // jacobian offset
      for (size_t i = 0; i < softmax_y; i++)
      {
        delta_val += error[err_off + i] * jacobian[jacob_off + i];
      }

      delta[x_id * softmax_y + y_id] = delta_val;
    }
  }

  Softmax::Softmax() : output(), delta()
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

    dim3 block_size_1(BLOCK_SIZE);
    dim3 grid_size_1(ceil(input.dim.x / BLOCK_SIZE));

    device_row_max_values<<<grid_size_1, block_size_1>>>(
      input.get_device_pointer(), &row_max[0], input.dim.x, input.dim.y);

    dim3 block_size_2(DIM_SIZE_2D, DIM_SIZE_2D - (DIM_SIZE_2D % input.dim.y));
    dim3 grid_size_2(ceil((input.dim.x * input.dim.y)
      / (block_size_2.x * block_size_2.y)));

    device_softmax<<<grid_size_2, block_size_2>>>(input.get_device_pointer(),
      &row_max[0], &row_sums[0], output.get_device_pointer(), input.dim.x,
      input.dim.y);

    cudaFree(row_max);
    cudaFree(row_sums);

    return output;
  }

  Neurons& Softmax::back_prop(Neurons& error, float learning_rate)
  {
    delta.allocate_memory(output.dim);

    float* jacobian = nullptr;
    cudaMalloc((void**)&jacobian,
      output.dim.x * output.dim.y * output.dim.y * sizeof(float));
    cudaMemset(jacobian, 0,
      output.dim.x * output.dim.y * output.dim.y * sizeof(float));

    dim3 block_size_1(DIM_SIZE_3D, DIM_SIZE_3D, DIM_SIZE_3D);
    dim3 grid_size_1(ceil((delta.dim.x * delta.dim.y)
      / (block_size_1.x * block_size_1.y * block_size_1.z)));

    device_softmax_jacobian<<<grid_size_1, block_size_1>>>(
      output.get_device_pointer(), &jacobian[0], output.dim.x, output.dim.y,
      output.dim.y * output.dim.y);

    dim3 block_size_2(DIM_SIZE_2D, DIM_SIZE_2D);
    dim3 grid_size_2(ceil((delta.dim.x * delta.dim.y) / BLOCK_SIZE));

    device_softmax_error<<<grid_size_2, block_size_2>>>(
      error.get_device_pointer(), &jacobian[0], delta.get_device_pointer(),
      output.dim.x, output.dim.y, output.dim.y * output.dim.y);

    cudaFree(jacobian);

    return delta;
  }

}
