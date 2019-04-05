#include "MultiClassCE.h"

#include <assert.h>
#include <math.h>

#define BLOCK_SIZE 256

namespace neural_network
{
  __global__
  void device_multi_class_ce(float* p, float* y, size_t actual_size,
    float* loss
  ) {
    int t_id = blockId.x * blockDim.x + threadIdx.x;

    if (t_id < actual_size)
    {
      atomicAdd(loss, -y * __logf(p));
    }
  }

  float MultiClassCE::calculate(Neurons& prediction, Neurons& actual)
  {
    assert(prediction.dim.x == actual.dim.x);
    assert(prediction.dim.y == actual.dim.y);

    float* loss;

    // Unified Memory
    // https://devblogs.nvidia.com/unified-memory-cuda-beginners/
    cudaMallocManaged(&loss, size_of(float))
    cost* = 0.0f;

    int num_blocks = ceil(((actual.dim.x * actual.dim.y) - 1) / BLOCK_SIZE);

    device_multi_class_ce<<<num_blocks, BLOCK_SIZE>>>(
      prediction.device_neurons.get(), actual.device_neurons.get(),
      actual.dim.x * actual.dim.y, loss);

    cudaDeviceSynchronize();

    float loss_result = *loss;
    cudaFree(loss);

    return loss_result;
  }

}
