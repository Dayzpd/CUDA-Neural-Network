#include "Neurons.h"

namespace neural_network
{

  Neurons::Neurons() : allocated(false)
  {

  }

  Neurons::Neurons(size_t x, size_t y) : dim(x, y), allocated(false)
  {
    allocate_memory();
  }

  Neurons::Neurons(Dim dim) : Neurons(dim.x, dim.y)
  {

  }

  // Concrete allocation primarily used for weights and biases that have fixed
  // user-defined sizes.
  void allocate_memory()
  {
    if (!this->allocated)
    {
      this->host_data = thrust::host_vector<float>(dim.x * this->dim.y);
      this->device_data = thrust::device_vector<float>(dim.x * dim.y);
      this->allocated = true;
    }
  }

  // On the fly allocation primarily used to allocate layer outputs.
  void allocate_memory(Dim dim)
  {
    if (!this->allocated)
    {
      this->dim = dim;
      this->host_data = thrust::host_vector<float>(dim.x * dim.y);
      this->device_data = thrust::device_vector<float>(dim.x * dim.y);
      this->allocated = true;
    }
  }

  void allocate_memory(size_t x, size_t y) : allocate_memory(Dim(x, y))
  {

  }

  void Neurons::memcpy_host_to_device()
  {
    this->device = this->host;
  }

  void Neurons::memcpy_device_to_host()
  {
    this->host = this->device;
  }

  float* get_device_pointer()
  {
    return thrust::raw_pointer_cast(&this->device[0]);
  }

}
