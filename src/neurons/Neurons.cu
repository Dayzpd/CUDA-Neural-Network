#include "Neurons.h"

#include <stdexcept>
#include <string>

namespace neural_network
{

  Neurons::Neurons(size_t x, size_t y) : dim(x, y),
    is_host_allocated(false), host_memory(nullptr), is_device_allocated(false),
    device_memory(nullptr)
  {

  }

  Neurons::Neurons(Dim dim) : Neurons(dim.x, dim.y)
  {

  }

  void Neurons::reserve_host_memory()
  {
    host_neurons = std::shared_ptr<float>(new float[dim.x * dim.y],
  		[&](float* ptr){ delete[] ptr; });

    is_host_allocated = true;
  }

  void Neurons::reserve_device_memory()
  {
    float* cuda_memory = nullptr;

    cudaMalloc(&cuda_memory, dim.x * dim.y * sizeof(float));
    check_cuda_error("CUDA Error (memory allocation)", __FILE__, __LINE__);

    device_neurons = std::shared_ptr<float>(cuda_memory,
  		[&](float* ptr){ cudaFree(ptr); });

    is_device_allocated = true;
  }

  void Neurons::reserve_memory()
  {
    if (!is_host_allocated)
    {
      reserve_host_memory();
    }

    if (!is_device_allocated)
    {
      reserve_device_memory();
    }
  }

  void Neurons::memcpy_host_to_device()
  {
    if (is_host_allocated && is_device_allocated)
    {
      cudaMemcpy(device_neurons.get(), host_neurons.get(),
        dim.x * dim.y * dim.z * sizeof(float), cudaMemcpyHostToDevice);
      check_cuda_error("CUDA Error (memory copy)", __FILE__, __LINE__);
    }
    else
    {
      throw runtime_error("Error: Reserve memory space before copying data.");
    }
  }

  void Neurons::memcpy_device_to_host()
  {
    if (is_host_allocated && is_device_allocated)
    {
      cudaMemcpy(host_neurons.get(), device_neurons.get(),
        dim.x * dim.y * dim.z * sizeof(float), cudaMemcpyDeviceToHost);
      check_cuda_error("CUDA Error (memory copy)", __FILE__, __LINE__);
    }
    else
    {
      throw runtime_error("Error: Reserve memory space before copying data.");
    }
  }

  float& Neurons::operator[](const int index)
  {
    return host_memory.get()[index];
  }

  const float& Neurons::operator[](const int index) const
  {
    return host_memory.get()[index];
  }

}
