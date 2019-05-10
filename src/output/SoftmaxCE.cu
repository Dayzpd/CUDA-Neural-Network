#include "SoftmaxCE.h"

#include <math.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

#define DIM_SIZE 32
#define BLOCK_SIZE 1024

__global__
void device_cross_entropy(float* p, float* y, float* loss, size_t actual_x,
  size_t actual_y
) {
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (t_id < actual_x * actual_y && y[t_id] == 1)
  {
    float loss_val = -__logf(p[t_id]) / (float)actual_x;
    atomicAdd(loss, loss_val);
  }
}

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
void device_softmax_ce_deriv(float* p, float* y, float* delta, size_t delta_x,
  size_t delta_y
) {
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (t_id < delta_x * delta_y)
  {
    delta[t_id] = p[t_id] - y[t_id];
  }
}

SoftmaxCE::SoftmaxCE(std::string name, bool verbose) : name(name), verbose(verbose),
  output(), delta()
{

}

SoftmaxCE::~SoftmaxCE()
{

}

float SoftmaxCE::loss(Neurons& actual)
{
  assert(output.dim.x == actual.dim.x);
  assert(output.dim.y == actual.dim.y);

  float h_loss;
  float* d_loss;
  cudaMalloc(&d_loss, sizeof(float));
  cudaMemset(d_loss, 0.0f, sizeof(float));

  dim3 block_size(BLOCK_SIZE);
  dim3 grid_size((actual.dim.x * actual.dim.y + BLOCK_SIZE - 1) / BLOCK_SIZE);

  device_cross_entropy<<<grid_size, block_size>>>(output.get_device_pointer(),
    actual.get_device_pointer(), d_loss, actual.dim.x, actual.dim.y);

  cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_loss);

  return h_loss; // returns mean of loss for the batch
}

Neurons& SoftmaxCE::forward_prop(Neurons& input)
{
  this->input = input;
  output.allocate_memory(input.dim);

  thrust::device_vector<float> row_max(input.dim.x, -1.0f);
  float* row_max_ptr = thrust::raw_pointer_cast(row_max.data());

  thrust::device_vector<float> row_sums(input.dim.x, 0.0f);
  float* row_sums_ptr = thrust::raw_pointer_cast(row_sums.data());

  dim3 block_size_1(BLOCK_SIZE);
  dim3 grid_size_1((input.dim.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  device_row_max_values<<<grid_size_1, block_size_1>>>(
    input.get_device_pointer(), row_max_ptr, input.dim.x, input.dim.y);

  if (verbose)
  {
    thrust::host_vector<float> host_row_max = row_max;
    for (int x = 0; x < input.dim.x; x++)
    {
      std::cout << "ROW MAX[" << x << "] = " << host_row_max[x] << std::endl;
    }
    host_row_max.empty();
    host_row_max.shrink_to_fit();
  }

  dim3 block_size_2(DIM_SIZE, DIM_SIZE - (DIM_SIZE % input.dim.y));
  dim3 grid_size_2(
    ( (input.dim.x * input.dim.y) + (block_size_2.x * block_size_2.y) - 1 ) /
    (block_size_2.x * block_size_2.y)
  );

  device_softmax<<<grid_size_2, block_size_2>>>(input.get_device_pointer(),
    row_max_ptr, row_sums_ptr, output.get_device_pointer(), input.dim.x,
    input.dim.y);

  if (verbose)
  {
    thrust::host_vector<float> host_row_sums = row_sums;
    for (int x = 0; x < input.dim.x; x++)
    {
      std::cout << "ROW SUMS[" << x << "] = " << host_row_sums[x] << std::endl;
    }
    host_row_sums.empty();
    host_row_sums.shrink_to_fit();
  }

  row_max.empty();
  row_max.shrink_to_fit();

  row_sums.empty();
  row_sums.shrink_to_fit();

  return output;
}

Neurons& SoftmaxCE::back_prop(Neurons& actual)
{
  delta.allocate_memory(output.dim);

  dim3 block_size(BLOCK_SIZE);
  dim3 grid_size((delta.dim.x * delta.dim.y + BLOCK_SIZE - 1) / BLOCK_SIZE);

  device_softmax_ce_deriv<<<grid_size, block_size>>>(
    output.get_device_pointer(), actual.get_device_pointer(),
    delta.get_device_pointer(), delta.dim.x, delta.dim.y
  );

  return delta;
}
