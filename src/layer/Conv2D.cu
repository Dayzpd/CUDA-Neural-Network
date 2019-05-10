
#include "Conv2D.h"
#include "../neurons/Dim.h"

#include <iostream>
#include <math.h>
#include <random>

#include <thrust/system_error.h>

#define DIM_SIZE_3D 8
#define BLOCK_SIZE 512

__global__
void device_conv_forward(float* input, float* kernel, float* output,
  float bias, size_t batch_size, size_t input_dim_size, size_t input_size,
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
    float output_val = 0.f;

    for (int x = 0; x < kernel_dim; x++)
    {
      for (int y = 0; y < kernel_dim; y++)
      {
        output_val +=
          input[(int)top_left_index + x * (int)input_dim_size + y] * kernel[x * (int)kernel_dim + y];
      }
    }

    int out_index = feature_num * pow((int)output_dim, 2) + row_num * (int)output_dim + col_num;
    output[out_index] = output_val + bias;
  }
}

__global__
void device_conv_back(float* error, float* kernel, float* delta,
  size_t batch_size, size_t error_dim, size_t error_size, size_t delta_dim,
  size_t delta_size, size_t kernel_dim
) {
  int feature_num = blockIdx.x * blockDim.x + threadIdx.x;
  int col_num = blockIdx.y * blockDim.y + threadIdx.y;
  int row_num = blockIdx.z * blockDim.z + threadIdx.z;

  if (feature_num < batch_size && row_num < delta_dim && col_num < delta_dim)
  {
    float delta_val = 0.f;

    // flip kernel by 180 degrees
    int rot_180 = kernel_dim - 1;
    int i = row_num - kernel_dim + 1;
    int j = col_num - kernel_dim + 1;

    for (int x = 0; x < kernel_dim; x++)
    {
      if (i >= 0 && i < error_dim)
      {
        for (int y = 0; y < kernel_dim; y++)
        {
          if (j >= 0 && j < error_dim)
          {
            delta_val +=
              error[feature_num * error_size + i * error_dim + j]
              * kernel[(rot_180 - x) * kernel_dim + rot_180 - y];
          }
          j += 1; // move to next column
        }
      }
      j = col_num - kernel_dim + 1;
      i += 1; // move to next row
    }

    int delta_index = feature_num * delta_size + row_num * delta_dim + col_num;
    delta[delta_index] = delta_val;
  }
}

__global__
void device_update_kernel(float* input, float* error, float* kernel,
  size_t batch_size, size_t input_dim, size_t input_size, size_t error_dim,
  size_t error_size, size_t kernel_dim, float learning_rate
) {
  int feature_num = blockIdx.x * blockDim.x + threadIdx.x;
  int col_num = blockIdx.y * blockDim.y + threadIdx.y; // kernel col
  int row_num = blockIdx.z * blockDim.z + threadIdx.z; // kernel row

  if (feature_num < batch_size && col_num < kernel_dim && row_num < kernel_dim)
  {
    int top_left_index = feature_num * input_size + row_num * input_dim + col_num;
    int rot_180 = error_dim - 1;
    float update_val = 0.f;
    float err_val;
    float input_val;
    for (int x = 0; x < error_dim; x++)
    {
      for (int y = 0; y < error_dim; y++)
      {
        err_val = error[feature_num * error_size + (rot_180 - x) * error_dim + rot_180 - y];

        if (top_left_index + x * input_dim + y >= batch_size * input_size)
        {
          break;
        }

        input_val = input[top_left_index + x * input_dim + y];

        //printf("top left: %d, x: %d, input_dim: %d, y: %d\n", top_left_index, x, (int)input_dim, y);

        update_val += err_val * input_val;
      }
    }

    atomicAdd(&kernel[row_num * kernel_dim + col_num],
      - learning_rate * (update_val / batch_size));
  }
}

__global__
void device_update_kernel_bias(float* error, float* bias, size_t batch_size,
  size_t error_size, float learning_rate
) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (x_id < error_size)
  {
    atomicAdd(bias, - learning_rate * (error[x_id] / batch_size));
  }
}

Conv2D::Conv2D(size_t kernel_dim, std::string name, bool verbose) :
  kernel(Dim(1, pow(kernel_dim, 2))), bias(0.01), kernel_dim(kernel_dim),
  name(name), verbose(verbose)
{
  kernel.allocate_memory();
  init_kernel();
}

Conv2D::~Conv2D()
{
  kernel.~Neurons();
}

void Conv2D::init_kernel()
{
  std::random_device rd;
  std::default_random_engine rand(rd());
  std::normal_distribution<float> norm(0.f, 0.5);

  for (size_t x = 0; x < kernel_dim * kernel_dim; x++)
  {
    kernel.host_data[x] = norm(rand);

    if (verbose)
    {
      printf("kernel[%d]: %f\n", (int)(x), kernel.host_data[x]);
    }
  }

  kernel.memcpy_host_to_device();
}

Neurons& Conv2D::forward_prop(Neurons& input)
{
  this->input = input;

  size_t input_dim_size = sqrt(input.dim.y);
  size_t output_dim = input_dim_size - kernel_dim + 1;
  output.allocate_memory(input.dim.x, pow(output_dim, 2));

  dim3 block_size(DIM_SIZE_3D, DIM_SIZE_3D, DIM_SIZE_3D);
  dim3 grid_size(
    (input.dim.x + DIM_SIZE_3D - 1) / DIM_SIZE_3D,
    (output_dim + DIM_SIZE_3D - 1) / DIM_SIZE_3D,
    (output_dim + DIM_SIZE_3D - 1) / DIM_SIZE_3D
  );

  device_conv_forward<<<grid_size, block_size>>>(
    input.get_device_pointer(), kernel.get_device_pointer(),
    output.get_device_pointer(), bias, input.dim.x, input_dim_size, input.dim.y,
    kernel_dim, output_dim);

  return output;
}

Neurons& Conv2D::back_prop(Neurons& error, float learning_rate)
{
  delta.allocate_memory(input.dim);

  backprop_error(error);

  update_kernel(error, learning_rate);

  update_bias(error, learning_rate);

  return delta;
}

void Conv2D::backprop_error(Neurons& error)
{
  size_t batch_size = delta.dim.x;
  size_t error_dim = sqrt(error.dim.y);
  size_t error_size = error.dim.y;
  size_t delta_dim = sqrt(delta.dim.y);
  size_t delta_size = delta.dim.y;
  size_t kernel_dim = sqrt(kernel.dim.y);

  dim3 block_size(DIM_SIZE_3D, DIM_SIZE_3D, DIM_SIZE_3D);
  dim3 grid_size(
    (batch_size + DIM_SIZE_3D - 1) / DIM_SIZE_3D,
    (delta_dim + DIM_SIZE_3D - 1) / DIM_SIZE_3D,
    (delta_dim + DIM_SIZE_3D - 1) / DIM_SIZE_3D
  );

  device_conv_back<<<grid_size, block_size>>>(
    error.get_device_pointer(), kernel.get_device_pointer(),
    delta.get_device_pointer(), batch_size, error_dim, error_size, delta_dim,
    delta_size, kernel_dim);
}

void Conv2D::update_kernel(Neurons& error, float learning_rate)
{
  size_t batch_size = input.dim.x;
  size_t error_dim = sqrt(error.dim.y);
  size_t error_size = error.dim.y;
  size_t input_dim = sqrt(input.dim.y);
  size_t input_size = input.dim.y;
  size_t kernel_dim = sqrt(kernel.dim.y);

  dim3 block_size(DIM_SIZE_3D, DIM_SIZE_3D, DIM_SIZE_3D);
  dim3 grid_size(
    (batch_size + DIM_SIZE_3D - 1) / DIM_SIZE_3D,
    (kernel_dim + DIM_SIZE_3D - 1) / DIM_SIZE_3D,
    (kernel_dim + DIM_SIZE_3D - 1) / DIM_SIZE_3D
  );

  device_update_kernel<<<grid_size, block_size>>>(input.get_device_pointer(),
    error.get_device_pointer(), kernel.get_device_pointer(), batch_size,
    input_dim, input_size, error_dim, error_size, kernel_dim, learning_rate);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    std::cerr << "Error (Conv2D::update_kernel[" << __LINE__ <<
      "]): Kernel launch failed.\n" <<
      "CUDA Error Message: " <<
      cudaGetErrorString(err) << std::endl;
    exit(-1);
  }
}

void Conv2D::update_bias(Neurons& error, float learning_rate)
{
  float h_bias;
  float* d_bias;
  cudaMalloc(&d_bias, sizeof(float));
  cudaMemset(d_bias, 0.f, sizeof(float));

  dim3 block_size(BLOCK_SIZE);
  dim3 grid_size((error.dim.x * error.dim.y + BLOCK_SIZE - 1) / BLOCK_SIZE);

  device_update_kernel_bias<<<grid_size, block_size>>>(
    error.get_device_pointer(), d_bias, error.dim.x, error.dim.x * error.dim.y,
    learning_rate
  );

  cudaMemcpy(&h_bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_bias);

  bias += h_bias;
}
