#include "Neurons.h"

namespace neural_network
{

  Neurons::Neurons(size_t x, size_t y) : dim(x, y), host(x * y),
    device(x * y)
  {

  }

  Neurons::Neurons(Dim dim) : Neurons(dim.x, dim.y)
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

  float* get_device_neurons()
  {
    return thrust::raw_pointer_cast(&this->device[0]);
  }

}
