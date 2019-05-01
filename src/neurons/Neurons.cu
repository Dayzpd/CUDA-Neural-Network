#include "Neurons.h"

namespace cuda_net
{

  Neurons::Neurons(size_t x, size_t y) : dim(x, y)
  {
    allocate_memory();
  }

  Neurons::Neurons(Dim dim) : Neurons(dim.x, dim.y)
  {

  }

  // Concrete allocation primarily used for weights and biases that have fixed
  // user-defined sizes.
  void Neurons::allocate_memory()
  {
    if (!allocated)
    {
      host_data = thrust::host_vector<float>(dim.x * dim.y);
      device_data = thrust::device_vector<float>(dim.x * dim.y);
      allocated = true;
    }
  }

  // On the fly allocation primarily used to allocate layer outputs.
  void Neurons::allocate_memory(Dim dim)
  {
    if (!allocated)
    {
      this->dim = dim;
      host_data = thrust::host_vector<float>(dim.x * dim.y);
      device_data = thrust::device_vector<float>(dim.x * dim.y);
      allocated = true;
    }
  }

  void Neurons::allocate_memory(size_t x, size_t y)
  {
    allocate_memory(Dim(x, y));
  }

  void Neurons::memcpy_host_to_device()
  {
    device_data = host_data;
  }

  void Neurons::memcpy_device_to_host()
  {
    host_data = device_data;
  }

  float* Neurons::get_device_pointer()
  {
    return thrust::raw_pointer_cast(&device_data[0]);
  }

}
