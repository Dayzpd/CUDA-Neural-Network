#include "CrossEntropy.h"

#include <assert.h>
#include <math.h>

#define BLOCK_SIZE 512

namespace neural_network
{
  __global__
  void device_cross_entropy(float* p, float* y, size_t actual_size,
    float* loss
  ) {
    int t_id = blockId.x * blockDim.x + threadIdx.x;

    if (t_id < actual_size && y[t_id] == 1)
    {
      atomicSub(&loss, __logf(p[t_id]));
    }
  }

  __global__
  void device_cross_entropy_deriv(float* p, float* y, float* p_delta,
    size_t actual_size
  ) {
    int t_id = blockId.x * blockDim.x + threadIdx.x;

    if (t_id < actual_size)
    {
      p_delta[t_id] = -(y[t_id] / p[t_id]);
    }
  }

  float CrossEntropy::calculate(Neurons& prob, Neurons& actual)
  {
    assert(prob.dim.x == actual.dim.x);
    assert(prob.dim.y == actual.dim.y);

    size_t actual_size = actual.dim.x * actual.dim.y;

    float* loss;

    // Unified Memory
    // https://devblogs.nvidia.com/unified-memory-cuda-beginners/
    cudaMallocManaged(&loss, size_of(float))
    loss* = 0.0f;

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(ceil(actual_size / BLOCK_SIZE));

    device_cross_entropy<<<grid_size, block_size>>>(prob.get_device_pointer(),
      actual.get_device_pointer(), loss, actual_size);

    float loss_result = *loss;
    cudaFree(loss);

    return loss_result / prob.dim.x; // returns mean of loss for the batch
  }

  Neurons CrossEntropy::calculate_deriv(Neurons& prob, Neurons& actual,
    Neurons& prob_delta
  ) {
    assert(prob.dim.x == actual.dim.x);
    assert(prob.dim.y == actual.dim.y);

    size_t actual_size = actual.dim.x * actual.dim.y;

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(ceil(actual_size / BLOCK_SIZE));

    device_cross_entropy_deriv<<<grid_size, block_size>>>(
      prob.get_device_pointer(), actual.get_device_pointer(),
      prob_delta.get_device_pointer(), actual_size);

    return prob_delta;
  }

}
